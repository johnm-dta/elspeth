# tests/unit/engine/orchestrator/test_export.py
"""Tests for post-run export and schema reconstruction functions.

export.py has two responsibilities:
1. Export audit trail to JSON/CSV sinks after run completion
2. Reconstruct Pydantic schemas from JSON schema dicts (for pipeline resume)

The export functions mock the LandscapeExporter, RecorderFactory, and sinks.
The schema reconstruction functions are pure logic — no mocks needed.
"""

from __future__ import annotations

import ast
import csv
import json
from contextlib import contextmanager
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import ValidationError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.engine.orchestrator.export import (
    _export_csv_multifile,
    _write_json_export_batches,
    export_landscape,
)
from elspeth.engine.orchestrator.schema_reconstruction import _json_schema_to_python_type, reconstruct_schema_from_json


@contextmanager
def _noop_track_operation(*_args: Any, **_kwargs: Any) -> Any:
    """Stub for track_operation that wires ctx.operation_id without touching the DB."""
    yield SimpleNamespace(operation_id="op-export")


class _CallRecorder:
    def __init__(self, return_value: Any = None, side_effect: BaseException | None = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            raise self.side_effect
        return self.return_value

    @property
    def call_args(self) -> SimpleNamespace:
        assert self.calls
        args, kwargs = self.calls[-1]
        return SimpleNamespace(args=args, kwargs=kwargs)

    def assert_called_once(self) -> None:
        assert len(self.calls) == 1

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        assert self.calls == [(args, kwargs)]

    def assert_not_called(self) -> None:
        assert self.calls == []


class _SinkDouble:
    name = "export_sink"
    plugin_version = "test"
    source_file_hash = "0" * 64

    def __init__(self, *, config: dict[str, Any] | None = None, **overrides: Any) -> None:
        self.config = config or {}
        self.node_id = None
        self.on_start = _CallRecorder()
        self.write = _CallRecorder()
        self.flush = _CallRecorder()
        self.on_complete = _CallRecorder()
        self.close = _CallRecorder()
        for k, v in overrides.items():
            setattr(self, k, v)


class _ExporterDouble:
    def __init__(self, grouped: dict[str, list[dict[str, Any]]]) -> None:
        self.export_run_grouped = _CallRecorder(return_value=grouped)
        self.iter_run_records_by_type = _CallRecorder(return_value=self._iter_run_records_by_type(grouped))

    @staticmethod
    def _iter_run_records_by_type(grouped: dict[str, list[dict[str, Any]]]) -> Any:
        for record_type, records in grouped.items():
            for record in records:
                yield record_type, record


def _make_settings(*, fmt: str = "json", sign: bool = False, sink: str = "output", include_raw_error_rows: bool = False) -> Any:
    return SimpleNamespace(
        landscape=SimpleNamespace(
            export=SimpleNamespace(
                format=fmt,
                sign=sign,
                sink=sink,
                include_raw_error_rows=include_raw_error_rows,
            )
        )
    )


def _make_sink_and_factory(*, config: dict[str, Any] | None = None, **overrides: Any) -> tuple[_SinkDouble, Any]:
    """Create a sink double and a factory that returns it."""
    sink = _SinkDouble(config=config, **overrides)
    for k, v in overrides.items():
        setattr(sink, k, v)
    return sink, lambda name: sink


@pytest.fixture(autouse=True)
def _mock_recorder_factory():
    """Prevent export tests from hitting real DB via RecorderFactory.register_node()."""
    with patch("elspeth.core.landscape.factory.RecorderFactory"):
        yield


# =============================================================================
# export_landscape — JSON format
# =============================================================================


class TestExportLandscapeJSON:
    """Tests for export_landscape with JSON format."""

    def _make_settings(self, *, fmt: str = "json", sign: bool = False, sink: str = "output", include_raw_error_rows: bool = False) -> Any:
        return _make_settings(fmt=fmt, sign=sign, sink=sink, include_raw_error_rows=include_raw_error_rows)

    def test_json_export_writes_records_to_sink(self) -> None:
        """JSON format exports all records through sink.write()."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            export_landscape(db, "run-1", settings, factory)

        sink.on_start.assert_called_once()
        sink.write.assert_called_once()
        sink.flush.assert_called_once()
        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()

    def test_json_export_skips_write_when_no_records(self) -> None:
        """Empty export produces no sink.write() call."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, factory)

        sink.write.assert_not_called()
        sink.flush.assert_called_once()
        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()

    def test_jsonl_export_writes_records_in_bounded_batches(self) -> None:
        """Large JSONL exports write bounded batches instead of one full-run list."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory(_format="jsonl")
        records = [{"record_type": "row", "index": i} for i in range(1001)]

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = records

            export_landscape(db, "run-1", settings, factory)

        assert [len(call[0][0]) for call in sink.write.calls] == [1000, 1]

    def test_json_array_export_writes_real_file_once_for_multi_batch_export(self, tmp_path: Path) -> None:
        """JSON array audit export publishes one final file, not one cumulative rewrite per batch."""
        from elspeth.plugins.sinks import json_sink as json_sink_module
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_path = tmp_path / "audit.json"
        sink = JSONSink({"path": str(output_path), "format": "json", "schema": {"mode": "observed"}})
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, payload_store=None, node_id="export:json")
        records = ({"record_type": "row", "index": i} for i in range(1001))
        replace_calls: list[tuple[Any, Any]] = []
        original_replace = json_sink_module.os.replace

        def counting_replace(src: Any, dst: Any) -> None:
            replace_calls.append((src, dst))
            original_replace(src, dst)

        with patch("elspeth.plugins.sinks.json_sink.os.replace", counting_replace):
            record_count, batches_written = _write_json_export_batches(
                sink=sink,
                ctx=ctx,
                records=records,
                batch_size=1000,
            )

        sink.close()

        assert record_count == 1001
        assert batches_written == 1
        assert len(replace_calls) == 1
        assert len(json.loads(output_path.read_text())) == 1001

    def test_json_export_records_count_as_operation_output(self) -> None:
        """JSON record_count is captured after streaming, not pre-counted as input."""
        db = object()
        settings = self._make_settings()
        _sink, factory = _make_sink_and_factory()
        captured: list[dict[str, Any]] = []

        @contextmanager
        def capture_track_operation(*_args: Any, **kwargs: Any) -> Any:
            handle = SimpleNamespace(operation_id="op-export", output_data=None)
            captured.append({"kwargs": kwargs, "handle": handle})
            yield handle

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", capture_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"record_type": "run"}, {"record_type": "node"}]

            export_landscape(db, "run-1", settings, factory)

        assert captured[0]["kwargs"]["input_data"] == {"export_format": "json"}
        assert captured[0]["handle"].output_data == {"record_count": 2, "batches_written": 1}

    def test_missing_sink_raises_valueerror(self) -> None:
        """Referencing non-existent sink raises clear error."""
        db = object()
        settings = self._make_settings(sink="nonexistent")

        def bad_factory(name: str) -> Any:
            raise ValueError(f"Export sink '{name}' not found in sink configuration")

        # LandscapeExporter constructs its own (module-level imported, so
        # autouse-unpatched) RecorderFactory, which now eagerly probes the
        # engine. Patch it so the bad-factory ValueError is what surfaces.
        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter"),
            pytest.raises(ValueError, match=r"nonexistent.*not found"),
        ):
            export_landscape(db, "run-1", settings, bad_factory)

    def test_signing_reads_env_key(self) -> None:
        """Signing enabled reads ELSPETH_SIGNING_KEY from env."""
        db = object()
        settings = self._make_settings(sign=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
            patch.dict("os.environ", {"ELSPETH_SIGNING_KEY": "test-key-123"}),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, factory)

        MockExporter.assert_called_once_with(db, signing_key=b"test-key-123", include_raw_error_rows=False)

    def test_raw_error_rows_opt_in_threads_to_exporter(self) -> None:
        """The export-config classification flag reaches the exporter
        (elspeth-384184c6ab): default False redacts raw failing rows from
        error records; the explicit opt-in restores them."""
        db: Any = object()
        settings = self._make_settings(include_raw_error_rows=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, factory)

        MockExporter.assert_called_once_with(db, signing_key=None, include_raw_error_rows=True)

    def test_signing_without_env_key_raises(self) -> None:
        """Signing enabled without ELSPETH_SIGNING_KEY raises ValueError."""
        db = object()
        settings = self._make_settings(sign=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            export_landscape(db, "run-1", settings, factory)

    @pytest.mark.parametrize("key_value", ["", "   "])
    def test_signing_with_empty_env_key_raises(self, key_value: str) -> None:
        """Signing enabled with an empty ELSPETH_SIGNING_KEY raises ValueError."""
        db = object()
        settings = self._make_settings(sign=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch.dict("os.environ", {"ELSPETH_SIGNING_KEY": key_value}, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            export_landscape(db, "run-1", settings, factory)

    def test_sink_close_called_when_write_raises(self) -> None:
        """sink.close() must be called even when sink.write() raises."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()
        sink.write.side_effect = RuntimeError("write failed")

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            with pytest.raises(RuntimeError, match="write failed"):
                export_landscape(db, "run-1", settings, factory)

        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()

    def test_sink_close_called_when_flush_raises(self) -> None:
        """sink.close() must be called even when sink.flush() raises."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()
        sink.flush.side_effect = RuntimeError("flush failed")

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            with pytest.raises(RuntimeError, match="flush failed"):
                export_landscape(db, "run-1", settings, factory)

        sink.write.assert_called_once()
        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()


# =============================================================================
# export_landscape — CSV format
# =============================================================================


class TestExportLandscapeCSV:
    """Tests for export_landscape with CSV format."""

    def _make_settings(self, *, sink: str = "output", sign: bool = False) -> Any:
        return _make_settings(fmt="csv", sign=sign, sink=sink)

    def test_csv_export_requires_path_in_sink_config(self) -> None:
        """CSV export needs file-based sink with 'path' config."""
        db = object()
        settings = self._make_settings()
        _sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter"),
            pytest.raises(ValueError, match="CSV export requires file-based sink"),
        ):
            export_landscape(db, "run-1", settings, factory)

    def test_csv_export_calls_multifile(self, tmp_path: Path) -> None:
        """CSV format delegates to _export_csv_multifile."""
        db = object()
        settings = self._make_settings()
        _sink, factory = _make_sink_and_factory(config={"path": str(tmp_path / "export.csv")})

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter"),
            patch("elspeth.engine.orchestrator.export._export_csv_multifile") as mock_csv,
        ):
            export_landscape(db, "run-1", settings, factory)

        mock_csv.assert_called_once()
        call_kwargs = mock_csv.call_args
        assert call_kwargs.kwargs["run_id"] == "run-1"

    def test_csv_export_uses_sink_lifecycle_around_multifile_writer(self, tmp_path: Path) -> None:
        """CSV export keeps sink lifecycle even when the writer owns multi-file output."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory(config={"path": str(tmp_path / "export.csv")})

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter"),
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
            patch("elspeth.engine.orchestrator.export._export_csv_multifile"),
        ):
            export_landscape(db, "run-1", settings, factory)

        sink.on_start.assert_called_once()
        sink.flush.assert_called_once()
        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()

    def test_export_landscape_does_not_read_csv_sink_path_directly(self) -> None:
        """Path extraction belongs behind the CSV audit-export writer boundary."""
        repo_root = Path(__file__).parents[4]
        path = repo_root / "src/elspeth/engine/orchestrator/export.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        export_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "export_landscape")

        offenders: list[str] = []
        for node in ast.walk(export_func):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "sink"
                and node.value.attr == "config"
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}:sink.config[...]")

        assert offenders == []


# =============================================================================
# _export_csv_multifile
# =============================================================================


class TestExportCSVMultifile:
    """Tests for the CSV multi-file export helper."""

    def test_creates_export_directory(self, tmp_path: Path) -> None:
        """Export creates the target directory."""
        export_dir = tmp_path / "audit_export"
        exporter = _ExporterDouble({})
        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
        )

        assert export_dir.exists()

    def test_strips_file_extension_from_path(self, tmp_path: Path) -> None:
        """If path has an extension, it's stripped (treated as directory name)."""
        export_path = tmp_path / "output.csv"
        exporter = _ExporterDouble({})
        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_path),
            sign=False,
        )

        # Directory should be "output" (no .csv extension)
        expected_dir = tmp_path / "output"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_writes_grouped_records_to_separate_files(self, tmp_path: Path) -> None:
        """Each record type gets its own CSV file."""
        export_dir = tmp_path / "export"

        exporter = _ExporterDouble(
            {
                "runs": [{"run_id": "r1", "status": "completed"}],
                "nodes": [
                    {"node_id": "n1", "type": "source"},
                    {"node_id": "n2", "type": "sink"},
                ],
            }
        )
        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
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

    def test_streams_csv_records_without_grouped_materializer(self, tmp_path: Path) -> None:
        """CSV export consumes typed records instead of whole-run grouped lists."""
        export_dir = tmp_path / "export"
        exporter = _ExporterDouble(
            {
                "rows": [
                    {"row_id": "r1", "source_data_hash": "h1"},
                    {"row_id": "r2", "source_data_hash": "h2"},
                ]
            }
        )
        exporter.export_run_grouped.side_effect = AssertionError("export_run_grouped materializes the full run")

        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
        )

        exporter.iter_run_records_by_type.assert_called_once_with("run-1", sign=False)
        with open(export_dir / "rows.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows == [
            {"row_id": "r1", "source_data_hash": "h1"},
            {"row_id": "r2", "source_data_hash": "h2"},
        ]

    def test_empty_record_types_skipped(self, tmp_path: Path) -> None:
        """Record types with empty lists don't produce files."""
        export_dir = tmp_path / "export"

        exporter = _ExporterDouble(
            {
                "runs": [{"run_id": "r1"}],
                "empty_type": [],
            }
        )
        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
        )

        assert (export_dir / "runs.csv").exists()
        assert not (export_dir / "empty_type.csv").exists()

    def test_csv_fieldnames_sorted_for_determinism(self, tmp_path: Path) -> None:
        """CSV headers are sorted alphabetically for deterministic output."""
        export_dir = tmp_path / "export"

        exporter = _ExporterDouble(
            {
                "data": [{"zebra": "z", "alpha": "a", "mid": "m"}],
            }
        )
        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
        )

        with open(export_dir / "data.csv") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == ["alpha", "mid", "zebra"]

    def test_union_of_all_keys_used_as_fieldnames(self, tmp_path: Path) -> None:
        """Records with different keys produce union of all keys as headers."""
        export_dir = tmp_path / "export"

        exporter = _ExporterDouble(
            {
                "mixed": [
                    {"common": "c1", "only_a": "a1"},
                    {"common": "c2", "only_b": "b1"},
                ],
            }
        )
        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
        )

        with open(export_dir / "mixed.csv") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert sorted(headers) == ["common", "only_a", "only_b"]

    def test_csv_cells_neutralize_spreadsheet_formula_prefixes(self, tmp_path: Path) -> None:
        """CSV audit export neutralizes untrusted strings that spreadsheets execute."""
        export_dir = tmp_path / "export"
        dangerous_values = {
            "row_data_json": '=HYPERLINK("https://example.test","click")',
            "error_details_json": "+SUM(1,2)",
            "negative": "-10+2",
            "mention": "@cmd",
            "tabbed": "\t=SUM(1,1)",
            "carriage_return": "\r=SUM(1,1)",
            "line_feed": "\n=SUM(1,1)",
        }

        exporter = _ExporterDouble(
            {
                "validation_errors": [
                    {
                        **dangerous_values,
                        "nested": {"message": '=cmd|"/C calc"!A0'},
                        "safe": "ordinary audit text",
                        "count": 3,
                    }
                ],
            }
        )
        _export_csv_multifile(
            exporter=exporter,
            run_id="run-1",
            artifact_path=str(export_dir),
            sign=False,
        )

        with open(export_dir / "validation_errors.csv", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        for field, value in dangerous_values.items():
            assert row[field] == f"'{value}"
        assert row["nested.message"] == '\'=cmd|"/C calc"!A0'
        assert row["safe"] == "ordinary audit text"
        assert row["count"] == "3"


def test_export_module_does_not_define_resume_schema_reconstruction_helpers() -> None:
    """Resume schema reconstruction lives outside the post-run export module."""
    repo_root = Path(__file__).parents[4]
    path = repo_root / "src/elspeth/engine/orchestrator/export.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    schema_helper_names = {
        "reconstruct_schema_from_json",
        "_create_schema_model",
        "_model_name_for_field",
        "_json_schema_to_python_type",
    }

    offenders = [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in schema_helper_names]

    assert offenders == []


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

    def test_nullable_decimal_anyof_pattern(self) -> None:
        """anyOf with number+string+null maps to Optional[Decimal].

        Regression: P1-2026-02-14 — Pydantic generates a 3-branch anyOf for
        Decimal | None: [{"type":"number"}, {"type":"string"}, {"type":"null"}].
        The old code only handled Decimal (2-branch, no null) and nullable
        (2-branch, one non-null), so this 3-branch pattern raised ValueError.
        """
        schema = {
            "properties": {
                "amount": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["amount"],
        }
        model = reconstruct_schema_from_json(schema)
        # Must accept Decimal values
        instance = model(amount=Decimal("99.99"))
        assert instance.amount == Decimal("99.99")
        # Must accept None (nullable)
        instance_none = model(amount=None)
        assert instance_none.amount is None

    def test_nullable_decimal_from_pydantic_model_json_schema(self) -> None:
        """Round-trip: Pydantic model with Decimal | None survives JSON schema reconstruction."""
        from pydantic import BaseModel

        class InvoiceRow(BaseModel):
            invoice_id: str
            amount: Decimal | None

        schema_dict = InvoiceRow.model_json_schema()
        model = reconstruct_schema_from_json(schema_dict)
        instance = model(invoice_id="INV-001", amount=Decimal("42.50"))
        assert instance.amount == Decimal("42.50")
        instance_none = model(invoice_id="INV-002", amount=None)
        assert instance_none.amount is None

    def test_unsupported_anyof_raises(self) -> None:
        """Unsupported anyOf pattern (e.g., Union[str, int]) raises."""
        schema = {
            "properties": {
                "weird": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["weird"],
        }
        with pytest.raises(AuditIntegrityError, match="unsupported anyOf"):
            reconstruct_schema_from_json(schema)


# =============================================================================
# reconstruct_schema_from_json — Error cases
# =============================================================================


class TestReconstructSchemaErrors:
    """Tests for error conditions in schema reconstruction."""

    def test_missing_properties_raises(self) -> None:
        """Schema without 'properties' key is malformed."""
        with pytest.raises(AuditIntegrityError, match="no 'properties'"):
            reconstruct_schema_from_json({"type": "object"})

    def test_empty_properties_without_additional_raises(self) -> None:
        """Empty properties without additionalProperties=true is invalid."""
        with pytest.raises(AuditIntegrityError, match="zero fields"):
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
        with pytest.raises(AuditIntegrityError, match=r"unsupported type.*custom_type"):
            reconstruct_schema_from_json(schema)

    def test_field_missing_type_raises(self) -> None:
        """Field without 'type' key (and no anyOf) raises."""
        schema = {
            "properties": {"x": {"description": "no type here"}},
            "required": ["x"],
        }
        with pytest.raises(AuditIntegrityError, match="no 'type'"):
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
