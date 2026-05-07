"""Tests for JSON source plugin."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.factories import make_context, make_source_context

# Dynamic schema config for tests - SourceDataConfig requires schema
DYNAMIC_SCHEMA = {"mode": "observed"}

# Standard quarantine routing for tests
QUARANTINE_SINK = "quarantine"


class TestJSONSource:
    """Tests for JSONSource plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_has_required_attributes(self) -> None:
        """JSONSource has name and output_schema."""
        from elspeth.plugins.sources.json_source import JSONSource

        assert JSONSource.name == "json"
        # output_schema is an instance attribute (set based on config)
        source = JSONSource(
            {
                "path": "/tmp/test.json",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
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

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        # All rows are SourceRow objects - access .row for data
        assert rows[0].row == {"id": 1, "name": "alice"}
        assert rows[0].is_quarantined is False
        assert rows[1].row["name"] == "bob"

    def test_load_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Load rows from JSONL file."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1, "name": "alice"}\n{"id": 2, "name": "bob"}\n{"id": 3, "name": "carol"}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[2].row["name"] == "carol"

    def test_auto_detect_jsonl_by_extension(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Auto-detect JSONL format from .jsonl extension."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"id": 2}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )  # No format specified
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_json_object_with_data_key(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Load rows from nested JSON object using data_key."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "wrapped.json"
        data = {
            "metadata": {"count": 2},
            "results": [{"id": 1}, {"id": 2}],
        }
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "data_key": "results",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": 1}

    def test_empty_lines_ignored_in_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Empty lines in JSONL are ignored."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "sparse.jsonl"
        jsonl_file.write_text('{"id": 1}\n\n{"id": 2}\n\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_file_not_found_raises(self, ctx: PluginContext) -> None:
        """Missing file raises FileNotFoundError."""
        from elspeth.plugins.sources.json_source import JSONSource

        source = JSONSource(
            {
                "path": "/nonexistent/file.json",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))

    def test_non_array_json_quarantined(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Non-array JSON is quarantined per Three-Tier Trust Model.

        External data (Tier 3) with wrong structure should be quarantined,
        not raise exceptions. This allows audit trail to record the failure.
        """
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "object.json"
        json_file.write_text('{"not": "an_array"}')

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        # Should NOT raise - should quarantine instead
        results = list(source.load(ctx))

        assert len(results) == 1
        assert results[0].is_quarantined is True
        assert results[0].quarantine_error is not None
        assert "array" in results[0].quarantine_error.lower()
        assert "dict" in results[0].quarantine_error.lower()

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_text("[]")

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        list(source.load(ctx))
        source.close()
        source.close()  # Should not raise

    def test_has_plugin_version(self) -> None:
        """JSONSource has explicit plugin_version for audit trail.

        Per CLAUDE.md auditability standard: every decision must be traceable
        to source data, configuration, AND code version. The plugin_version
        attribute is recorded in the Landscape audit trail's nodes table.
        """
        from elspeth.plugins.sources.json_source import JSONSource

        # Class attribute check - doesn't require valid config
        assert hasattr(JSONSource, "plugin_version")
        assert isinstance(JSONSource.plugin_version, str)
        assert JSONSource.plugin_version != "0.0.0"  # Must not be placeholder
        assert JSONSource.plugin_version == "1.0.0"


class TestJSONSourceFlexibleContract:
    """Regression tests for FLEXIBLE first-row infer-and-lock semantics."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_flexible_infers_extra_field_and_quarantines_type_drift(self, tmp_path: Path, ctx: PluginContext) -> None:
        """First valid row infers extras; subsequent rows with type drift are quarantined."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [
            {"id": 1, "extra": "alpha"},
            {"id": 2, "extra": 42},  # type drift: int vs inferred str
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {"mode": "flexible", "fields": ["id: int"]},
                "on_validation_failure": "quarantine",
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].is_quarantined is False
        assert rows[1].is_quarantined is True
        assert rows[1].quarantine_error is not None
        assert "extra" in rows[1].quarantine_error

        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.mode == "FLEXIBLE"
        assert contract.locked is True
        assert {field.normalized_name for field in contract.fields} == {"id", "extra"}

    def test_flexible_all_invalid_rows_still_publish_locked_declared_contract(
        self,
        tmp_path: Path,
        ctx: PluginContext,
    ) -> None:
        """All-invalid FLEXIBLE input keeps declared fields and locks contract."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [
            {"id": "bad"},
            {"id": "still_bad"},
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {"mode": "flexible", "fields": ["id: int"]},
                "on_validation_failure": "quarantine",
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(row.is_quarantined for row in rows)

        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.mode == "FLEXIBLE"
        assert contract.locked is True
        assert [field.normalized_name for field in contract.fields] == ["id"]


class TestJSONSourceConfigValidation:
    """Test JSONSource config validation."""

    def test_missing_path_raises_error(self) -> None:
        """Empty config raises PluginConfigError."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        with pytest.raises(PluginConfigError, match="path"):
            JSONSource({"schema": DYNAMIC_SCHEMA, "on_validation_failure": QUARANTINE_SINK})

    def test_empty_path_raises_error(self) -> None:
        """Empty path string raises PluginConfigError."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        with pytest.raises(PluginConfigError, match="path cannot be empty"):
            JSONSource(
                {
                    "path": "",
                    "schema": DYNAMIC_SCHEMA,
                    "on_validation_failure": QUARANTINE_SINK,
                }
            )

    def test_unknown_field_raises_error(self) -> None:
        """Unknown config field raises PluginConfigError."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        with pytest.raises(PluginConfigError, match="Extra inputs"):
            JSONSource(
                {
                    "path": "/tmp/test.json",
                    "schema": DYNAMIC_SCHEMA,
                    "on_validation_failure": QUARANTINE_SINK,
                    "unknown_field": "value",
                }
            )

    def test_missing_schema_raises_error(self) -> None:
        """Missing schema raises PluginConfigError."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        # Error message uses alias "schema" not field name "schema_config"
        with pytest.raises(PluginConfigError, match=r"schema[\s\S]*Field required"):
            JSONSource({"path": "/tmp/test.json", "on_validation_failure": QUARANTINE_SINK})

    def test_missing_on_validation_failure_raises_error(self) -> None:
        """Missing on_validation_failure raises PluginConfigError."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        with pytest.raises(PluginConfigError, match="on_validation_failure"):
            JSONSource({"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA})


class TestJSONSourceQuarantineYielding:
    """Tests for JSON source yielding SourceRow.quarantined() for invalid rows."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_invalid_row_yields_quarantined_source_row(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Invalid row yields SourceRow.quarantined() with error info."""
        import json

        from elspeth.contracts import SourceRow
        from elspeth.plugins.sources.json_source import JSONSource

        # JSON with invalid row (score is not an int)
        json_file = tmp_path / "data.json"
        data = [
            {"id": 1, "name": "alice", "score": 95},
            {"id": 2, "name": "bob", "score": "bad"},  # Invalid
            {"id": 3, "name": "carol", "score": 92},
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "on_validation_failure": "quarantine",
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str", "score: int"],
                },
            }
        )

        results = list(source.load(ctx))

        # 2 valid rows + 1 quarantined - all are SourceRow
        assert len(results) == 3
        assert all(isinstance(r, SourceRow) for r in results)

        # First and third are valid SourceRows
        assert results[0].is_quarantined is False
        assert results[0].row["name"] == "alice"
        assert results[2].is_quarantined is False
        assert results[2].row["name"] == "carol"

        # Second is a quarantined SourceRow
        quarantined = results[1]
        assert quarantined.is_quarantined is True
        assert quarantined.row["name"] == "bob"
        assert quarantined.row["score"] == "bad"  # Original value preserved
        assert quarantined.quarantine_destination == "quarantine"
        assert quarantined.quarantine_error is not None
        assert "score" in quarantined.quarantine_error  # Error mentions the field

    def test_discard_mode_does_not_yield_invalid_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        """When on_validation_failure='discard', invalid rows are not yielded."""
        import json

        from elspeth.contracts import SourceRow
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [
            {"id": 1, "name": "alice", "score": 95},
            {"id": 2, "name": "bob", "score": "bad"},  # Invalid
            {"id": 3, "name": "carol", "score": 92},
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "on_validation_failure": "discard",
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str", "score: int"],
                },
            }
        )

        results = list(source.load(ctx))

        # Only 2 valid rows - invalid row discarded
        assert len(results) == 2
        assert all(isinstance(r, SourceRow) and not r.is_quarantined for r in results)
        assert {r.row["name"] for r in results} == {"alice", "carol"}

    def test_jsonl_invalid_row_yields_quarantined(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL format also yields SourceRow.quarantined() for invalid rows."""
        from elspeth.contracts import SourceRow
        from elspeth.plugins.sources.json_source import JSONSource

        # JSONL with invalid row
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"id": 1, "name": "alice", "score": 95}\n{"id": 2, "name": "bob", "score": "bad"}\n{"id": 3, "name": "carol", "score": 92}\n'
        )

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "quarantine",
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str", "score: int"],
                },
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 3
        assert isinstance(results[1], SourceRow)
        assert results[1].is_quarantined is True
        assert results[1].row["name"] == "bob"


class TestJSONSourceParseErrors:
    """Tests for JSON source handling of parse/decode errors.

    Per CLAUDE.md Three-Tier Trust Model, external data (Tier 3) should be
    quarantined on parse errors, not crash the pipeline.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_jsonl_malformed_line_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Malformed JSONL line is quarantined, not crash the pipeline.

        This is the core bug: json.JSONDecodeError should be caught and
        the row quarantined, allowing subsequent valid lines to process.
        """
        from elspeth.contracts import SourceRow
        from elspeth.plugins.sources.json_source import JSONSource

        # JSONL with malformed line (line 2 is invalid JSON)
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"id": 1, "name": "alice"}\n'
            "{bad json\n"  # Malformed - missing quotes, colon, closing brace
            '{"id": 3, "name": "carol"}\n'
        )

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT raise - malformed line should be quarantined
        results = list(source.load(ctx))

        # All 3 lines should be processed: 2 valid + 1 quarantined
        assert len(results) == 3

        # First and third are valid
        assert results[0].is_quarantined is False
        assert results[0].row == {"id": 1, "name": "alice"}
        assert results[2].is_quarantined is False
        assert results[2].row == {"id": 3, "name": "carol"}

        # Second is quarantined with parse error info
        quarantined = results[1]
        assert isinstance(quarantined, SourceRow)
        assert quarantined.is_quarantined is True
        assert quarantined.quarantine_destination == "quarantine"
        assert quarantined.quarantine_error is not None
        assert "JSON" in quarantined.quarantine_error or "json" in quarantined.quarantine_error

    def test_jsonl_malformed_line_with_discard_mode(self, tmp_path: Path, ctx: PluginContext) -> None:
        """With on_validation_failure='discard', malformed lines are dropped silently."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1}\n{bad json\n{"id": 3}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "discard",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))

        # Only 2 valid rows - malformed line discarded
        assert len(results) == 2
        assert all(not r.is_quarantined for r in results)
        assert results[0].row == {"id": 1}
        assert results[1].row == {"id": 3}

    def test_jsonl_invalid_encoding_line_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Invalid-encoding JSONL line is quarantined, not crash the pipeline."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_bytes(b'{"id": 1}\n\xff\xfe\n{"id": 3}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "encoding": "utf-8",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 3
        assert results[0].is_quarantined is False
        assert results[0].row == {"id": 1}
        assert results[2].is_quarantined is False
        assert results[2].row == {"id": 3}

        quarantined = results[1]
        assert quarantined.is_quarantined is True
        assert quarantined.quarantine_error is not None
        assert "line 2" in quarantined.quarantine_error
        assert "utf-8" in quarantined.quarantine_error.lower()
        assert "__raw_bytes_hex__" in quarantined.row
        assert quarantined.row["__line_number__"] == 2

    def test_jsonl_utf16_surrogateescape_line_quarantined_not_crash(
        self,
        tmp_path: Path,
        ctx: PluginContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """UTF-16 surrogateescape lines are quarantined without crashing."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text("{}", encoding="utf-8")

        # Simulate text-decoder output containing surrogateescape bytes for a
        # UTF-16 source line. Re-encoding with utf-16 used to crash here.
        fake_file = io.StringIO('{"id": 1}\n\udcff\n{"id": 3}\n')
        real_open = open

        def fake_open(file: str | Path, *args: Any, **kwargs: Any) -> object:
            if Path(file) == jsonl_file and kwargs.get("encoding") == "utf-16":
                fake_file.seek(0)
                return fake_file
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "encoding": "utf-16",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 3
        assert results[0].is_quarantined is False
        assert results[0].row == {"id": 1}
        assert results[2].is_quarantined is False
        assert results[2].row == {"id": 3}

        quarantined = results[1]
        assert quarantined.is_quarantined is True
        assert quarantined.quarantine_error is not None
        assert "line 2" in quarantined.quarantine_error
        assert "utf-16" in quarantined.quarantine_error.lower()
        assert "__raw_bytes_hex__" in quarantined.row
        assert quarantined.row["__line_number__"] == 2

    def test_jsonl_invalid_encoding_line_with_discard_mode(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Invalid-encoding JSONL line is dropped when discard mode is configured."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_bytes(b'{"id": 1}\n\xff\xfe\n{"id": 3}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "encoding": "utf-8",
                "on_validation_failure": "discard",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))
        assert len(results) == 2
        assert all(not r.is_quarantined for r in results)
        assert results[0].row == {"id": 1}
        assert results[1].row == {"id": 3}

    def test_jsonl_utf16_records_load_without_decode_quarantine(self, tmp_path: Path, ctx: PluginContext) -> None:
        """UTF-16 JSONL should parse as valid rows (no raw-byte line splitting)."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "utf16-data.jsonl"
        jsonl_file.write_text('{"id": 1, "name": "alice"}\n{"id": 2, "name": "bob"}\n', encoding="utf-16")

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "encoding": "utf-16",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))
        assert len(results) == 2
        assert all(not row.is_quarantined for row in results)
        assert results[0].row == {"id": 1, "name": "alice"}
        assert results[1].row == {"id": 2, "name": "bob"}

    def test_jsonl_quarantined_row_contains_raw_line_data(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Quarantined parse error should contain the raw line for audit."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        malformed_line = "{totally broken: json here"
        jsonl_file.write_text(f'{{"id": 1}}\n{malformed_line}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))
        quarantined = results[1]

        # The quarantined row should contain the raw line for audit traceability
        # (since we couldn't parse it into a dict)
        assert quarantined.is_quarantined is True
        # Row data should include the raw line content
        assert "__raw_line__" in quarantined.row or malformed_line in str(quarantined.row)

    def test_json_array_malformed_file_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Malformed JSON array file should quarantine, not crash."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        # Invalid JSON: missing closing bracket
        json_file.write_text('[{"id": 1}, {"id": 2}')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should not crash - should yield quarantined row
        results = list(source.load(ctx))

        # Should get exactly 1 quarantined row (file-level failure)
        assert len(results) == 1
        quarantined = results[0]
        assert quarantined.is_quarantined is True
        assert quarantined.quarantine_destination == "quarantine"
        assert quarantined.quarantine_error is not None
        assert "JSON parse error" in quarantined.quarantine_error
        assert "file_path" in quarantined.row

    def test_json_array_malformed_file_with_discard_mode(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Malformed JSON array file with discard mode should not yield any rows."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": 1}, {"id": 2}')  # Invalid JSON

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "on_validation_failure": "discard",
                "schema": {"mode": "observed"},
            }
        )

        # Should not crash, should yield nothing
        results = list(source.load(ctx))
        assert len(results) == 0


class TestJSONSourceNonFiniteConstants:
    """Tests for JSON source rejection of NaN/Infinity constants.

    Per CLAUDE.md Three-Tier Trust Model and canonical JSON policy:
    - NaN/Infinity are non-standard JSON constants
    - Python's json module accepts them by default
    - They must be rejected at the source boundary (Tier 3)
    - Rows with these values should be quarantined, not crash downstream

    Bug: P2-2026-01-21-jsonsource-nonfinite-constants-allowed
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_jsonl_nan_constant_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL with NaN constant is quarantined at parse time.

        NaN is a non-standard JSON constant that Python's json module accepts
        by default. It must be rejected at the source boundary to prevent
        downstream crashes in canonical hashing.
        """
        from elspeth.contracts import SourceRow
        from elspeth.plugins.sources.json_source import JSONSource

        # JSONL with NaN constant (non-standard JSON)
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"id": 1, "name": "alice"}\n'
            '{"id": 2, "score": NaN}\n'  # Non-standard: NaN constant
            '{"id": 3, "name": "carol"}\n'
        )

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT raise - NaN should be quarantined at parse time
        results = list(source.load(ctx))

        # All 3 lines should be processed: 2 valid + 1 quarantined
        assert len(results) == 3

        # First and third are valid
        assert results[0].is_quarantined is False
        assert results[0].row == {"id": 1, "name": "alice"}
        assert results[2].is_quarantined is False
        assert results[2].row == {"id": 3, "name": "carol"}

        # Second is quarantined with NaN rejection error
        quarantined = results[1]
        assert isinstance(quarantined, SourceRow)
        assert quarantined.is_quarantined is True
        assert quarantined.quarantine_destination == "quarantine"
        assert quarantined.quarantine_error is not None
        # Error should mention NaN or non-standard constant
        assert "NaN" in quarantined.quarantine_error or "non-standard" in quarantined.quarantine_error.lower()

    def test_jsonl_infinity_constant_quarantined(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL with Infinity constant is quarantined at parse time."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"id": 1, "value": 42}\n'
            '{"id": 2, "value": Infinity}\n'  # Non-standard: Infinity constant
            '{"id": 3, "value": -Infinity}\n'  # Non-standard: -Infinity constant
        )

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))

        # 1 valid + 2 quarantined
        assert len(results) == 3
        assert results[0].is_quarantined is False
        assert results[1].is_quarantined is True
        assert results[2].is_quarantined is True

        # Both Infinity values should be rejected
        assert results[1].quarantine_error is not None
        assert "Infinity" in results[1].quarantine_error or "non-standard" in results[1].quarantine_error.lower()

    def test_json_array_nan_constant_quarantined(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSON array with NaN constant is quarantined at parse time."""
        from elspeth.plugins.sources.json_source import JSONSource

        # Note: We write raw text since Python's json.dumps would convert NaN to null
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": 1, "score": NaN}]')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))

        # File-level quarantine since NaN is in the whole array parse
        assert len(results) == 1
        assert results[0].is_quarantined is True
        assert results[0].quarantine_error is not None
        assert "NaN" in results[0].quarantine_error or "non-standard" in results[0].quarantine_error.lower()

    def test_nan_with_discard_mode_not_yielded(self, tmp_path: Path, ctx: PluginContext) -> None:
        """With on_validation_failure='discard', NaN rows are dropped silently."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1, "value": 42}\n{"id": 2, "score": NaN}\n{"id": 3, "value": 100}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "discard",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))

        # Only 2 valid rows - NaN row discarded
        assert len(results) == 2
        assert all(not r.is_quarantined for r in results)
        assert results[0].row == {"id": 1, "value": 42}
        assert results[1].row == {"id": 3, "value": 100}

    def test_nan_quarantine_contains_raw_line(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Quarantined NaN row should contain raw line for audit traceability."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        nan_line = '{"id": 2, "score": NaN}'
        jsonl_file.write_text(f'{{"id": 1}}\n{nan_line}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))
        quarantined = results[1]

        # The quarantined row should contain the raw line for audit traceability
        assert quarantined.is_quarantined is True
        assert "__raw_line__" in quarantined.row
        assert nan_line in quarantined.row["__raw_line__"]


class TestJSONSourceDataKeyStructuralErrors:
    """Tests for JSON source handling of data_key structural mismatches.

    Per CLAUDE.md Three-Tier Trust Model (Tier 3 - external data):
    - If data_key is configured but JSON root is a list, quarantine don't crash
    - If data_key is configured but key doesn't exist in JSON object, quarantine
    - If data_key extraction results in non-list, quarantine don't crash

    Bug: P2-2026-01-21-jsonsource-data-key-missing-crash
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_data_key_on_list_root_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """data_key configured but JSON root is a list - quarantine, not TypeError.

        This tests the case where an API changes from returning {results: [...]}
        to returning [...] directly. Should quarantine gracefully.
        """
        from elspeth.plugins.sources.json_source import JSONSource

        # JSON file is a list, but data_key expects an object
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": 1}, {"id": 2}]')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "data_key": "results",  # Expects object with "results" key
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT raise TypeError - should quarantine
        results = list(source.load(ctx))

        # Should yield 1 quarantined row for the structural error
        assert len(results) == 1
        assert results[0].is_quarantined is True
        assert results[0].quarantine_destination == "quarantine"
        # Error should indicate structural mismatch
        error = results[0].quarantine_error
        assert error is not None
        assert "list" in error.lower() or "dict" in error.lower() or "object" in error.lower()

    def test_data_key_missing_in_object_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """data_key configured but key doesn't exist in JSON object - quarantine, not KeyError.

        This tests the case where data_key: "results" is configured but
        the JSON has {"items": [...]} instead.
        """
        from elspeth.plugins.sources.json_source import JSONSource

        # JSON object has "items" key, not "results"
        json_file = tmp_path / "data.json"
        json_file.write_text('{"items": [{"id": 1}, {"id": 2}], "metadata": {}}')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "data_key": "results",  # Key doesn't exist
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT raise KeyError - should quarantine
        results = list(source.load(ctx))

        # Should yield 1 quarantined row for the missing key
        assert len(results) == 1
        assert results[0].is_quarantined is True
        assert results[0].quarantine_destination == "quarantine"
        # Error should indicate missing key
        error = results[0].quarantine_error
        assert error is not None
        assert "results" in error or "key" in error.lower() or "missing" in error.lower()

    def test_data_key_extracts_non_list_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """data_key extracts a non-list value - quarantine, not crash.

        This tests the case where data_key points to a scalar or nested object
        instead of an array.
        """
        from elspeth.plugins.sources.json_source import JSONSource

        # data_key points to an object, not a list
        json_file = tmp_path / "data.json"
        json_file.write_text('{"results": {"count": 0, "items": []}, "status": "ok"}')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "data_key": "results",  # Points to object, not list
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT raise ValueError - should quarantine
        results = list(source.load(ctx))

        # Should yield 1 quarantined row for wrong type
        assert len(results) == 1
        assert results[0].is_quarantined is True
        # Error should indicate type mismatch (expected array)
        error = results[0].quarantine_error
        assert error is not None
        assert "array" in error.lower() or "list" in error.lower() or "dict" in error.lower()

    def test_data_key_structural_error_with_discard_mode(self, tmp_path: Path, ctx: PluginContext) -> None:
        """With on_validation_failure='discard', structural errors yield nothing."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": 1}]')  # List root, but data_key expects object

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "data_key": "results",
                "on_validation_failure": "discard",  # Don't yield quarantined
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT raise - should discard silently
        results = list(source.load(ctx))

        # No rows yielded - structural error discarded
        assert len(results) == 0

    def test_data_key_structural_error_records_validation_error(self, tmp_path: Path) -> None:
        """Structural errors are recorded via ctx.record_validation_error().

        With a real Landscape recorder, the validation error is persisted
        to the database. This test verifies the recording path succeeds.
        """
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.plugins.sources.json_source import JSONSource
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(source_plugin_name="json")
        ctx = PluginContext(
            run_id=setup.run_id,
            node_id=setup.source_node_id,
            config={},
            landscape=setup.factory.plugin_audit_writer(),
        )

        json_file = tmp_path / "data.json"
        json_file.write_text('{"wrong_key": [{"id": 1}]}')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "data_key": "results",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))

        # Structural error yields a quarantined SourceRow
        assert len(results) == 1
        assert results[0].quarantine_destination == "quarantine"
        assert results[0].quarantine_error is not None
        assert "results" in results[0].quarantine_error  # The missing data_key

        # Verify validation error was recorded to Landscape via factory's data_flow repo
        errors = setup.factory.data_flow.get_validation_errors_for_run(ctx.run_id)
        assert len(errors) == 1
        assert "results" in errors[0].error  # The missing data_key

    def test_data_key_structural_error_uses_parse_schema_mode(self, tmp_path: Path) -> None:
        """Structural data_key errors record contract-valid schema_mode='parse'."""
        from unittest.mock import MagicMock

        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_text('{"wrong_key": [{"id": 1}]}')

        mock_landscape = MagicMock()
        mock_landscape.record_validation_error.return_value = "verr_test"
        ctx = make_context(
            run_id="test-run",
            node_id="source_json",
            landscape=mock_landscape,
        )

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "data_key": "results",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        list(source.load(ctx))

        call_kwargs = mock_landscape.record_validation_error.call_args[1]
        assert call_kwargs["schema_mode"] == "parse"


class TestJSONSourceArrayModeUnicodeDecodeError:
    """Regression tests for P1: JSONSource crashes on invalid byte sequences in array mode.

    Bug: P1-2026-02-14-jsonsource-crashes-on-invalid-byte-sequences-in-json-array-mode

    Per Three-Tier Trust Model, external data with invalid encoding must be
    quarantined, not crash the pipeline. JSONL mode already handled this;
    JSON array mode did not.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_invalid_utf8_bytes_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Invalid UTF-8 byte sequences in JSON array file are quarantined, not crash."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        # Write raw bytes: valid JSON prefix followed by invalid UTF-8 bytes
        json_file.write_bytes(b'[{"id": 1}, {"name": "\xff\xfe"}]')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "encoding": "utf-8",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT crash with UnicodeDecodeError — should quarantine
        results = list(source.load(ctx))

        assert len(results) == 1
        quarantined = results[0]
        assert quarantined.is_quarantined is True
        assert quarantined.quarantine_destination == "quarantine"
        assert quarantined.quarantine_error is not None
        assert "utf-8" in quarantined.quarantine_error.lower()
        assert "file_path" in quarantined.row

    def test_invalid_utf8_bytes_with_discard_mode(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Invalid UTF-8 bytes in array mode with discard yields nothing."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_bytes(b'[{"id": 1}, {"name": "\xff\xfe"}]')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "encoding": "utf-8",
                "on_validation_failure": "discard",
                "schema": {"mode": "observed"},
            }
        )

        results = list(source.load(ctx))
        assert len(results) == 0

    def test_truncated_utf16_file_quarantined_not_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Truncated UTF-16 file in JSON array mode is quarantined, not crash."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        # Write a valid UTF-16 BOM then truncate mid-sequence
        json_file.write_bytes(b'\xff\xfe[{"id": 1}')  # Incomplete UTF-16

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "encoding": "utf-16",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should NOT crash — should quarantine
        results = list(source.load(ctx))

        # Either UnicodeDecodeError or JSONDecodeError depending on decoder —
        # both should be quarantined, not crash
        assert len(results) >= 1
        assert all(r.is_quarantined for r in results)

    def test_invalid_encoding_records_validation_error(self, tmp_path: Path) -> None:
        """Invalid encoding in array mode records validation error in audit trail."""
        from unittest.mock import MagicMock

        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_bytes(b'[{"id": 1}, {"name": "\xff\xfe"}]')

        mock_landscape = MagicMock()
        mock_landscape.record_validation_error.return_value = "verr_test"
        ctx = make_context(
            run_id="test-run",
            node_id="source_json",
            landscape=mock_landscape,
        )

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "encoding": "utf-8",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        list(source.load(ctx))

        # Verify validation error was recorded via landscape
        mock_landscape.record_validation_error.assert_called_once()
        call_kwargs = mock_landscape.record_validation_error.call_args[1]
        assert call_kwargs["schema_mode"] == "parse"
        assert "utf-8" in call_kwargs["error"].lower()

    def test_pure_binary_garbage_quarantined(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Completely invalid binary file is quarantined in array mode."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_bytes(bytes(range(128, 256)))  # All high bytes

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "encoding": "utf-8",
                "on_validation_failure": "quarantine",
                "schema": {"mode": "observed"},
            }
        )

        # Should quarantine the decode error, not crash
        results = list(source.load(ctx))

        assert len(results) == 1
        assert results[0].is_quarantined is True
        assert results[0].quarantine_error is not None


class TestJSONSourceRowShapeBoundaryErrors:
    """Regression tests for malformed JSON row shapes at the source boundary."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_jsonl_scalar_row_quarantined_and_later_rows_continue(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL scalar rows are quarantined instead of crashing the source."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "scalar-first.jsonl"
        jsonl_file.write_text('1\n{"id": 2}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 2
        assert results[0].is_quarantined is True
        assert results[0].quarantine_error is not None
        assert "expected json object" in results[0].quarantine_error.lower()
        assert "int" in results[0].quarantine_error.lower()
        assert results[1].is_quarantined is False
        assert results[1].row == {"id": 2}

    def test_json_array_scalar_element_quarantined_and_later_rows_continue(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Array-mode scalar elements are quarantined instead of aborting the file."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "mixed.json"
        json_file.write_text('[1, {"id": 2}]')

        source = JSONSource(
            {
                "path": str(json_file),
                "format": "json",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 2
        assert results[0].is_quarantined is True
        assert results[0].quarantine_error is not None
        assert "expected json object" in results[0].quarantine_error.lower()
        assert "int" in results[0].quarantine_error.lower()
        assert results[1].is_quarantined is False
        assert results[1].row == {"id": 2}

    def test_later_row_normalization_collision_quarantined_and_later_rows_continue(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Normalization collisions quarantine the offending row instead of crashing."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "colliding-keys.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"A B": 2, "a_b": 3}\n{"id": 4}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 3
        assert results[0].is_quarantined is False
        assert results[0].row == {"id": 1}
        assert results[1].is_quarantined is True
        assert results[1].quarantine_error is not None
        assert "collision after normalization" in results[1].quarantine_error.lower()
        assert results[1].row == {"A B": 2, "a_b": 3}
        assert results[2].is_quarantined is False
        assert results[2].row == {"id": 4}

    def test_row_key_normalizing_to_empty_identifier_quarantined(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Keys that normalize to an empty identifier are quarantined, not fatal."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "empty-normalized-key.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"!!!": 2}\n{"id": 4}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "format": "jsonl",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 3
        assert results[0].is_quarantined is False
        assert results[1].is_quarantined is True
        assert results[1].quarantine_error is not None
        assert "normalizes to empty string" in results[1].quarantine_error.lower()
        assert results[1].row == {"!!!": 2}
        assert results[2].is_quarantined is False
        assert results[2].row == {"id": 4}


class TestJSONSourceKeyNormalization:
    """Tests for JSON source field name normalization and resolution.

    JSON keys may contain spaces, mixed case, or other characters that
    are not valid Python identifiers. The source normalizes these at
    the Tier 3 boundary.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with proper FK records for validation error recording."""
        return make_source_context(plugin_name="json")

    def test_nontrivial_key_normalization(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSON keys with spaces and mixed case are normalized to snake_case identifiers."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [
            {"Customer Name": "Alice", "Order ID": 101},
            {"Customer Name": "Bob", "Order ID": 102},
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(not r.is_quarantined for r in rows)
        # Keys should be normalized to snake_case
        assert rows[0].row == {"customer_name": "Alice", "order_id": 101}
        assert rows[1].row == {"customer_name": "Bob", "order_id": 102}

    def test_field_mapping_override(self, tmp_path: Path, ctx: PluginContext) -> None:
        """field_mapping config overrides default normalization for specified keys."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [
            {"Customer Name": "Alice", "Order ID": 101},
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
                "field_mapping": {"customer_name": "client_name"},
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert not rows[0].is_quarantined
        # "Customer Name" uses field_mapping override, "Order ID" uses default normalization
        assert rows[0].row["client_name"] == "Alice"
        assert rows[0].row["order_id"] == 101

    def test_get_field_resolution_returns_mapping(self, tmp_path: Path, ctx: PluginContext) -> None:
        """get_field_resolution() returns the resolution mapping after load()."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [{"Customer Name": "Alice", "Order ID": 101}]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )

        # Before load, field resolution is None
        assert source.get_field_resolution() is None

        list(source.load(ctx))

        resolution = source.get_field_resolution()
        assert resolution is not None
        mapping, version = resolution
        # Mapping contains original->normalized entries
        assert mapping["Customer Name"] == "customer_name"
        assert mapping["Order ID"] == "order_id"
        assert version is not None

    def test_first_row_quarantined_key_rebuild(self, tmp_path: Path, ctx: PluginContext) -> None:
        """When first row is quarantined and second row has different keys, resolution rebuilds.

        If the first row's keys produce a stale resolution mapping, subsequent rows
        with different keys should trigger a rebuild so normalization is correct.
        """
        from elspeth.plugins.sources.json_source import JSONSource

        # First object fails FIXED schema validation (missing required 'score' field),
        # second object succeeds but has an extra key not in the first row.
        json_file = tmp_path / "data.json"
        data = [
            {"id": 1, "name": "alice"},  # Missing 'score' -- quarantined by FLEXIBLE schema
            {"id": 2, "name": "bob", "score": 95, "Extra Field": "bonus"},  # Valid + extra key
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {
                    "mode": "flexible",
                    "fields": ["id: int", "name: str", "score: int"],
                },
                "on_validation_failure": "quarantine",
            }
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        # First row quarantined (missing required 'score')
        assert rows[0].is_quarantined is True
        # Second row valid -- "Extra Field" should be normalized
        assert rows[1].is_quarantined is False
        assert rows[1].row["extra_field"] == "bonus"
        assert rows[1].row["score"] == 95

    def test_fixed_schema_fast_path_sets_contract_in_init(self, tmp_path: Path) -> None:
        """FIXED schema sets contract immediately in __init__ without waiting for first row."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_text("[]")  # Empty -- we only test __init__ behavior

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str"],
                },
                "on_validation_failure": "quarantine",
            }
        )

        # Contract should be set immediately (fast path), not deferred
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True
        assert contract.mode == "FIXED"
        # ContractBuilder should be None (not needed for FIXED)
        assert source._contract_builder is None
