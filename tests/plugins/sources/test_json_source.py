"""Tests for JSON source plugin."""

import json
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol

# Dynamic schema config for tests - SourceDataConfig requires schema
DYNAMIC_SCHEMA = {"fields": "dynamic"}

# Standard quarantine routing for tests
QUARANTINE_SINK = "quarantine"


class TestJSONSource:
    """Tests for JSONSource plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """JSONSource implements SourceProtocol."""
        from elspeth.plugins.sources.json_source import JSONSource

        source = JSONSource(
            {
                "path": "/tmp/test.json",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        assert isinstance(source, SourceProtocol)

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

    def test_non_array_json_raises(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Non-array JSON raises ValueError."""
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
        with pytest.raises(ValueError, match="Expected JSON array"):
            list(source.load(ctx))

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


class TestJSONSourceConfigValidation:
    """Test JSONSource config validation."""

    def test_missing_path_raises_error(self) -> None:
        """Empty config raises PluginConfigError."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        with pytest.raises(PluginConfigError, match="path"):
            JSONSource({"schema": DYNAMIC_SCHEMA, "on_validation_failure": QUARANTINE_SINK})

    def test_empty_path_raises_error(self) -> None:
        """Empty path string raises PluginConfigError."""
        from elspeth.plugins.config_base import PluginConfigError
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
        from elspeth.plugins.config_base import PluginConfigError
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
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        with pytest.raises(PluginConfigError, match=r"require.*schema"):
            JSONSource({"path": "/tmp/test.json", "on_validation_failure": QUARANTINE_SINK})

    def test_missing_on_validation_failure_raises_error(self) -> None:
        """Missing on_validation_failure raises PluginConfigError."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.sources.json_source import JSONSource

        with pytest.raises(PluginConfigError, match="on_validation_failure"):
            JSONSource({"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA})


class TestJSONSourceQuarantineYielding:
    """Tests for JSON source yielding SourceRow.quarantined() for invalid rows."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

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
                    "mode": "strict",
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
                    "mode": "strict",
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
                    "mode": "strict",
                    "fields": ["id: int", "name: str", "score: int"],
                },
            }
        )

        results = list(source.load(ctx))

        assert len(results) == 3
        assert isinstance(results[1], SourceRow)
        assert results[1].is_quarantined is True
        assert results[1].row["name"] == "bob"
