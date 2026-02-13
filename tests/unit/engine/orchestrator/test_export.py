# tests/unit/engine/orchestrator/test_export.py
"""Tests for post-run export and schema reconstruction functions.

export.py has two responsibilities:
1. Export audit trail to JSON/CSV sinks after run completion
2. Reconstruct Pydantic schemas from JSON schema dicts (for pipeline resume)

The export functions mock the LandscapeExporter and sinks.
The schema reconstruction functions are pure logic — no mocks needed.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from pydantic import ValidationError

from elspeth.engine.orchestrator.export import (
    _export_csv_multifile,
    _json_schema_to_python_type,
    export_landscape,
    reconstruct_schema_from_json,
)
from elspeth.plugins.protocols import SinkProtocol

# =============================================================================
# export_landscape — JSON format
# =============================================================================


class TestExportLandscapeJSON:
    """Tests for export_landscape with JSON format."""

    def _make_settings(self, *, fmt: str = "json", sign: bool = False, sink: str = "output") -> Mock:
        settings = Mock()
        settings.landscape.export.format = fmt
        settings.landscape.export.sign = sign
        settings.landscape.export.sink = sink
        return settings

    def test_json_export_writes_records_to_sink(self) -> None:
        """JSON format exports all records through sink.write()."""
        db = Mock()
        settings = self._make_settings()
        sink = Mock()
        sink.config = {}
        sinks: dict[str, SinkProtocol] = {"output": sink}

        with patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter:
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            export_landscape(db, "run-1", settings, sinks)

        sink.write.assert_called_once()
        sink.flush.assert_called_once()
        sink.close.assert_called_once()

    def test_json_export_skips_write_when_no_records(self) -> None:
        """Empty export produces no sink.write() call."""
        db = Mock()
        settings = self._make_settings()
        sink = Mock()
        sink.config = {}
        sinks: dict[str, SinkProtocol] = {"output": sink}

        with patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter:
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, sinks)

        sink.write.assert_not_called()
        sink.flush.assert_called_once()
        sink.close.assert_called_once()

    def test_missing_sink_raises_valueerror(self) -> None:
        """Referencing non-existent sink raises clear error."""
        db = Mock()
        settings = self._make_settings(sink="nonexistent")
        sinks: dict[str, SinkProtocol] = {"output": Mock()}

        with pytest.raises(ValueError, match=r"nonexistent.*not found"):
            export_landscape(db, "run-1", settings, sinks)

    def test_signing_reads_env_key(self) -> None:
        """Signing enabled reads ELSPETH_SIGNING_KEY from env."""
        db = Mock()
        settings = self._make_settings(sign=True)
        sink = Mock()
        sink.config = {}
        sinks: dict[str, SinkProtocol] = {"output": sink}

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch.dict("os.environ", {"ELSPETH_SIGNING_KEY": "test-key-123"}),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, sinks)

        MockExporter.assert_called_once_with(db, signing_key=b"test-key-123")

    def test_signing_without_env_key_raises(self) -> None:
        """Signing enabled without ELSPETH_SIGNING_KEY raises ValueError."""
        db = Mock()
        settings = self._make_settings(sign=True)
        sinks: dict[str, SinkProtocol] = {"output": Mock()}

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            export_landscape(db, "run-1", settings, sinks)

    def test_sink_close_called_when_write_raises(self) -> None:
        """sink.close() must be called even when sink.write() raises."""
        db = Mock()
        settings = self._make_settings()
        sink = Mock()
        sink.config = {}
        sink.write.side_effect = RuntimeError("write failed")
        sinks: dict[str, SinkProtocol] = {"output": sink}

        with patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter:
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            with pytest.raises(RuntimeError, match="write failed"):
                export_landscape(db, "run-1", settings, sinks)

        sink.close.assert_called_once()

    def test_sink_close_called_when_flush_raises(self) -> None:
        """sink.close() must be called even when sink.flush() raises."""
        db = Mock()
        settings = self._make_settings()
        sink = Mock()
        sink.config = {}
        sink.flush.side_effect = RuntimeError("flush failed")
        sinks: dict[str, SinkProtocol] = {"output": sink}

        with patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter:
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            with pytest.raises(RuntimeError, match="flush failed"):
                export_landscape(db, "run-1", settings, sinks)

        sink.write.assert_called_once()
        sink.close.assert_called_once()


# =============================================================================
# export_landscape — CSV format
# =============================================================================


class TestExportLandscapeCSV:
    """Tests for export_landscape with CSV format."""

    def _make_settings(self, *, sink: str = "output", sign: bool = False) -> Mock:
        settings = Mock()
        settings.landscape.export.format = "csv"
        settings.landscape.export.sign = sign
        settings.landscape.export.sink = sink
        return settings

    def test_csv_export_requires_path_in_sink_config(self) -> None:
        """CSV export needs file-based sink with 'path' config."""
        db = Mock()
        settings = self._make_settings()
        sink = Mock()
        sink.config = {}  # No 'path' key
        sinks: dict[str, SinkProtocol] = {"output": sink}

        with pytest.raises(ValueError, match="CSV export requires file-based sink"):
            export_landscape(db, "run-1", settings, sinks)

    def test_csv_export_calls_multifile(self, tmp_path: Path) -> None:
        """CSV format delegates to _export_csv_multifile."""
        db = Mock()
        settings = self._make_settings()
        sink = Mock()
        sink.config = {"path": str(tmp_path / "export.csv")}
        sinks: dict[str, SinkProtocol] = {"output": sink}

        with patch("elspeth.engine.orchestrator.export._export_csv_multifile") as mock_csv:
            export_landscape(db, "run-1", settings, sinks)

        mock_csv.assert_called_once()
        call_kwargs = mock_csv.call_args
        assert call_kwargs.kwargs["run_id"] == "run-1"


# =============================================================================
# _export_csv_multifile
# =============================================================================


class TestExportCSVMultifile:
    """Tests for the CSV multi-file export helper."""

    def test_creates_export_directory(self, tmp_path: Path) -> None:
        """Export creates the target directory."""
        export_dir = tmp_path / "audit_export"
        exporter = Mock()
        exporter.export_run_grouped.return_value = {}
        ctx = Mock()

        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
            ctx=ctx,
        )

        assert export_dir.exists()

    def test_strips_file_extension_from_path(self, tmp_path: Path) -> None:
        """If path has an extension, it's stripped (treated as directory name)."""
        export_path = tmp_path / "output.csv"
        exporter = Mock()
        exporter.export_run_grouped.return_value = {}
        ctx = Mock()

        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_path),
            sign=False,
            ctx=ctx,
        )

        # Directory should be "output" (no .csv extension)
        expected_dir = tmp_path / "output"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_writes_grouped_records_to_separate_files(self, tmp_path: Path) -> None:
        """Each record type gets its own CSV file."""
        export_dir = tmp_path / "export"

        # Mock formatter to pass through dicts
        with patch("elspeth.core.landscape.formatters.CSVFormatter") as MockFormatter:
            formatter = MockFormatter.return_value
            formatter.format.side_effect = lambda r: r  # Pass through

            exporter = Mock()
            exporter.export_run_grouped.return_value = {
                "runs": [{"run_id": "r1", "status": "completed"}],
                "nodes": [
                    {"node_id": "n1", "type": "source"},
                    {"node_id": "n2", "type": "sink"},
                ],
            }
            ctx = Mock()

            _export_csv_multifile(
                exporter=exporter,
                run_id="run-1",
                artifact_path=str(export_dir),
                sign=False,
                ctx=ctx,
            )

        # Check files exist
        assert (export_dir / "runs.csv").exists()
        assert (export_dir / "nodes.csv").exists()

        # Verify runs.csv content
        with open(export_dir / "runs.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "r1"

        # Verify nodes.csv content
        with open(export_dir / "nodes.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_empty_record_types_skipped(self, tmp_path: Path) -> None:
        """Record types with empty lists don't produce files."""
        export_dir = tmp_path / "export"

        with patch("elspeth.core.landscape.formatters.CSVFormatter") as MockFormatter:
            formatter = MockFormatter.return_value
            formatter.format.side_effect = lambda r: r

            exporter = Mock()
            exporter.export_run_grouped.return_value = {
                "runs": [{"run_id": "r1"}],
                "empty_type": [],
            }
            ctx = Mock()

            _export_csv_multifile(
                exporter=exporter,
                run_id="run-1",
                artifact_path=str(export_dir),
                sign=False,
                ctx=ctx,
            )

        assert (export_dir / "runs.csv").exists()
        assert not (export_dir / "empty_type.csv").exists()

    def test_csv_fieldnames_sorted_for_determinism(self, tmp_path: Path) -> None:
        """CSV headers are sorted alphabetically for deterministic output."""
        export_dir = tmp_path / "export"

        with patch("elspeth.core.landscape.formatters.CSVFormatter") as MockFormatter:
            formatter = MockFormatter.return_value
            formatter.format.side_effect = lambda r: r

            exporter = Mock()
            exporter.export_run_grouped.return_value = {
                "data": [{"zebra": "z", "alpha": "a", "mid": "m"}],
            }
            ctx = Mock()

            _export_csv_multifile(
                exporter=exporter,
                run_id="run-1",
                artifact_path=str(export_dir),
                sign=False,
                ctx=ctx,
            )

        with open(export_dir / "data.csv") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == ["alpha", "mid", "zebra"]

    def test_union_of_all_keys_used_as_fieldnames(self, tmp_path: Path) -> None:
        """Records with different keys produce union of all keys as headers."""
        export_dir = tmp_path / "export"

        with patch("elspeth.core.landscape.formatters.CSVFormatter") as MockFormatter:
            formatter = MockFormatter.return_value
            formatter.format.side_effect = lambda r: r

            exporter = Mock()
            exporter.export_run_grouped.return_value = {
                "mixed": [
                    {"common": "c1", "only_a": "a1"},
                    {"common": "c2", "only_b": "b1"},
                ],
            }
            ctx = Mock()

            _export_csv_multifile(
                exporter=exporter,
                run_id="run-1",
                artifact_path=str(export_dir),
                sign=False,
                ctx=ctx,
            )

        with open(export_dir / "mixed.csv") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert sorted(headers) == ["common", "only_a", "only_b"]


# =============================================================================
# reconstruct_schema_from_json — Primitive types
# =============================================================================


class TestReconstructSchemaBasic:
    """Tests for reconstruct_schema_from_json with basic types."""

    def test_string_field(self) -> None:
        """String type maps correctly."""
        schema = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(name="Alice")
        assert instance.name == "Alice"

    def test_integer_field(self) -> None:
        """Integer type maps correctly."""
        schema = {
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(count=42)
        assert instance.count == 42

    def test_number_field(self) -> None:
        """Number type maps to float."""
        schema = {
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(score=3.14)
        assert instance.score == pytest.approx(3.14)

    def test_boolean_field(self) -> None:
        """Boolean type maps correctly."""
        schema = {
            "properties": {"active": {"type": "boolean"}},
            "required": ["active"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(active=True)
        assert instance.active is True

    def test_optional_field_defaults_to_none(self) -> None:
        """Fields not in 'required' default to None."""
        schema = {
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(name="Bob")
        assert instance.name == "Bob"
        assert instance.age is None

    def test_all_fields_required(self) -> None:
        """All fields in required list are enforced."""
        schema = {
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(a="x", b=1)
        assert instance.a == "x"
        assert instance.b == 1

    def test_no_required_key_means_all_optional(self) -> None:
        """Missing 'required' key treats all fields as optional."""
        schema = {
            "properties": {"name": {"type": "string"}},
        }
        model = reconstruct_schema_from_json(schema)
        instance = model()
        assert instance.name is None

    def test_array_field(self) -> None:
        """Array type maps to list."""
        schema = {
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

    def test_array_field_with_items_enforces_item_type(self) -> None:
        """Array with items schema enforces item type on resume."""
        schema = {
            "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
            "required": ["scores"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(scores=[1, 2, 3])
        assert instance.scores == [1, 2, 3]

        with pytest.raises(ValidationError, match="scores\\.0"):
            model(scores=["not-an-int"])

    def test_object_field(self) -> None:
        """Object type maps to dict."""
        schema = {
            "properties": {"metadata": {"type": "object"}},
            "required": ["metadata"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(metadata={"key": "val"})
        assert instance.metadata == {"key": "val"}

    def test_object_field_with_properties_enforces_nested_schema(self) -> None:
        """Nested object properties are reconstructed and validated."""
        schema = {
            "properties": {
                "profile": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                    "required": ["age", "name"],
                }
            },
            "required": ["profile"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(profile={"age": 42, "name": "Ada"})
        assert instance.profile.age == 42
        assert instance.profile.name == "Ada"

        with pytest.raises(ValidationError, match="profile\\.age"):
            model(profile={"age": "not-an-int", "name": "Ada"})

        with pytest.raises(ValidationError, match="profile\\.name"):
            model(profile={"age": 42})


# =============================================================================
# reconstruct_schema_from_json — Format specifiers
# =============================================================================


class TestReconstructSchemaFormats:
    """Tests for string format specifiers (datetime, date, UUID, etc.)."""

    def test_datetime_format(self) -> None:
        """date-time format maps to datetime."""
        schema = {
            "properties": {"ts": {"type": "string", "format": "date-time"}},
            "required": ["ts"],
        }
        model = reconstruct_schema_from_json(schema)
        now = datetime.now(tz=UTC)
        instance = model(ts=now)
        assert instance.ts == now

    def test_date_format(self) -> None:
        """date format maps to date."""
        schema = {
            "properties": {"d": {"type": "string", "format": "date"}},
            "required": ["d"],
        }
        model = reconstruct_schema_from_json(schema)
        today = datetime.now(tz=UTC).date()
        instance = model(d=today)
        assert instance.d == today

    def test_time_format(self) -> None:
        """time format maps to time."""
        schema = {
            "properties": {"t": {"type": "string", "format": "time"}},
            "required": ["t"],
        }
        model = reconstruct_schema_from_json(schema)
        now = time(12, 30, 0)
        instance = model(t=now)
        assert instance.t == now

    def test_duration_format(self) -> None:
        """duration format maps to timedelta."""
        schema = {
            "properties": {"dur": {"type": "string", "format": "duration"}},
            "required": ["dur"],
        }
        model = reconstruct_schema_from_json(schema)
        td = timedelta(hours=1, minutes=30)
        instance = model(dur=td)
        assert instance.dur == td

    def test_uuid_format(self) -> None:
        """uuid format maps to UUID."""
        schema = {
            "properties": {"id": {"type": "string", "format": "uuid"}},
            "required": ["id"],
        }
        model = reconstruct_schema_from_json(schema)
        uid = UUID("12345678-1234-5678-1234-567812345678")
        instance = model(id=uid)
        assert instance.id == uid

    def test_unknown_format_treated_as_string(self) -> None:
        """Unknown format falls back to str."""
        schema = {
            "properties": {"custom": {"type": "string", "format": "custom-fmt"}},
            "required": ["custom"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(custom="hello")
        assert instance.custom == "hello"


# =============================================================================
# reconstruct_schema_from_json — anyOf patterns
# =============================================================================


class TestReconstructSchemaAnyOf:
    """Tests for anyOf patterns (Decimal, nullable)."""

    def test_decimal_anyof_pattern(self) -> None:
        """anyOf with number+string maps to Decimal."""
        schema = {
            "properties": {
                "amount": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
            "required": ["amount"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(amount=Decimal("123.45"))
        assert instance.amount == Decimal("123.45")

    def test_nullable_type_pattern(self) -> None:
        """anyOf with type+null maps to Optional[type] — accepts both values and None."""
        schema = {
            "properties": {
                "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(name="Alice")
        assert instance.name == "Alice"
        # Nullable fields must also accept None
        instance_none = model(name=None)
        assert instance_none.name is None

    def test_nullable_datetime_pattern(self) -> None:
        """anyOf with datetime+null resolves to Optional[datetime]."""
        schema = {
            "properties": {
                "ts": {
                    "anyOf": [
                        {"type": "string", "format": "date-time"},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["ts"],
        }
        model = reconstruct_schema_from_json(schema)
        now = datetime.now(tz=UTC)
        instance = model(ts=now)
        assert instance.ts == now
        # Nullable datetime must accept None
        instance_none = model(ts=None)
        assert instance_none.ts is None

    def test_unsupported_anyof_raises(self) -> None:
        """Unsupported anyOf pattern (e.g., Union[str, int]) raises."""
        schema = {
            "properties": {
                "weird": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["weird"],
        }
        with pytest.raises(ValueError, match="unsupported anyOf"):
            reconstruct_schema_from_json(schema)


# =============================================================================
# reconstruct_schema_from_json — Error cases
# =============================================================================


class TestReconstructSchemaErrors:
    """Tests for error conditions in schema reconstruction."""

    def test_missing_properties_raises(self) -> None:
        """Schema without 'properties' key is malformed."""
        with pytest.raises(ValueError, match="no 'properties'"):
            reconstruct_schema_from_json({"type": "object"})

    def test_empty_properties_without_additional_raises(self) -> None:
        """Empty properties without additionalProperties=true is invalid."""
        with pytest.raises(ValueError, match="zero fields"):
            reconstruct_schema_from_json({"properties": {}})

    def test_empty_properties_with_additional_creates_dynamic(self) -> None:
        """Empty properties + additionalProperties=true creates dynamic schema."""
        schema = {"properties": {}, "additionalProperties": True}
        model = reconstruct_schema_from_json(schema)
        # Dynamic schema should accept arbitrary fields
        instance = model(any_field="value", another=42)
        assert instance.any_field == "value"

    def test_unsupported_type_raises(self) -> None:
        """Unknown JSON schema type (e.g., 'custom') raises."""
        schema = {
            "properties": {"x": {"type": "custom_type"}},
            "required": ["x"],
        }
        with pytest.raises(ValueError, match=r"unsupported type.*custom_type"):
            reconstruct_schema_from_json(schema)

    def test_field_missing_type_raises(self) -> None:
        """Field without 'type' key (and no anyOf) raises."""
        schema = {
            "properties": {"x": {"description": "no type here"}},
            "required": ["x"],
        }
        with pytest.raises(ValueError, match="no 'type'"):
            reconstruct_schema_from_json(schema)


# =============================================================================
# _json_schema_to_python_type — Direct tests
# =============================================================================


class TestJsonSchemaToPythonType:
    """Direct tests for the type mapping helper."""

    @pytest.mark.parametrize(
        ("field_info", "expected_type"),
        [
            ({"type": "string"}, str),
            ({"type": "integer"}, int),
            ({"type": "number"}, float),
            ({"type": "boolean"}, bool),
            ({"type": "array"}, list),
            ({"type": "object"}, dict),
            ({"type": "string", "format": "date-time"}, datetime),
            ({"type": "string", "format": "date"}, date),
            ({"type": "string", "format": "time"}, time),
            ({"type": "string", "format": "duration"}, timedelta),
            ({"type": "string", "format": "uuid"}, UUID),
        ],
        ids=[
            "string",
            "integer",
            "number",
            "boolean",
            "array",
            "object",
            "datetime",
            "date",
            "time",
            "duration",
            "uuid",
        ],
    )
    def test_type_mapping(self, field_info: dict[str, Any], expected_type: type) -> None:
        """Each JSON schema type maps to the correct Python type."""
        result = _json_schema_to_python_type("test_field", field_info)
        assert result is expected_type

    def test_decimal_anyof(self) -> None:
        """Decimal pattern recognized via anyOf."""
        field_info = {"anyOf": [{"type": "number"}, {"type": "string"}]}
        assert _json_schema_to_python_type("price", field_info) is Decimal

    def test_nullable_resolves_to_optional(self) -> None:
        """Nullable pattern resolves to Optional[inner_type]."""
        field_info = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        result = _json_schema_to_python_type("count", field_info)
        # Should be int | None (UnionType), not bare int
        assert result == int | None

    def test_nullable_ref_resolves_through_defs(self) -> None:
        """Nullable $ref pattern (Optional[NestedModel]) resolves via $defs.

        Regression: Pydantic emits Optional[NestedModel] as:
          {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}]}
        The $ref entry has NO "type" key, so filtering on item["type"]
        raised KeyError.
        """
        import types

        from pydantic import BaseModel, create_model

        field_info: dict[str, Any] = {
            "anyOf": [
                {"$ref": "#/$defs/Address"},
                {"type": "null"},
            ],
        }
        schema_defs: dict[str, Any] = {
            "Address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["city", "zip"],
            }
        }
        # Must not raise KeyError — the $ref item lacks a "type" key
        result = _json_schema_to_python_type(
            "address",
            field_info,
            schema_defs=schema_defs,
            create_model=create_model,
            schema_base=BaseModel,
        )
        # Should be Optional[AddressModel] — a UnionType containing the model and None
        assert isinstance(result, types.UnionType)
        type_args = result.__args__
        assert type(None) in type_args
        # The non-None arg should be a Pydantic model subclass
        model_type = next(t for t in type_args if t is not type(None))
        assert issubclass(model_type, BaseModel)
        instance = model_type(city="London", zip="SW1A 1AA")
        assert instance.city == "London"

    def test_nullable_ref_full_schema_roundtrip(self) -> None:
        """Full schema with Optional[NestedModel] field reconstructs correctly.

        Regression: reconstruct_schema_from_json crashed on schemas containing
        fields like Optional[Address] because the anyOf filter accessed
        item["type"] on $ref entries that have no "type" key.
        """
        schema: dict[str, Any] = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                }
            },
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "anyOf": [
                        {"$ref": "#/$defs/Address"},
                        {"type": "null"},
                    ],
                },
            },
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        # Non-null value
        instance = model(name="Alice", address={"city": "London"})
        assert instance.name == "Alice"
        assert instance.address.city == "London"
        # Null value (optional)
        instance2 = model(name="Bob", address=None)
        assert instance2.address is None
