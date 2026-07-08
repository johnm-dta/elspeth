"""Post-run audit export functions.

This module handles:
- Exporting the Landscape audit trail to JSON or CSV format after run completion

The export functions support two modes:
- JSON format: All records written to a single sink (handles heterogeneous records)
- CSV format: Separate files per record type (CSV requires homogeneous schemas)

Resume schema reconstruction lives in schema_reconstruction.py. This module
keeps compatibility re-exports for older import paths.

IMPORTANT: Import Cycle Prevention
----------------------------------
This module uses TYPE_CHECKING for imports that would cause cycles.
LandscapeDB is typed but imported conditionally to avoid circular imports.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol
    from elspeth.core.config import ElspethSettings
    from elspeth.core.landscape import LandscapeDB

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_context import PluginContext
from elspeth.core.operations import track_operation
from elspeth.engine.orchestrator.schema_reconstruction import (
    _create_schema_model as _create_schema_model,
)
from elspeth.engine.orchestrator.schema_reconstruction import (
    _json_schema_to_python_type as _json_schema_to_python_type,
)
from elspeth.engine.orchestrator.schema_reconstruction import (
    _model_name_for_field as _model_name_for_field,
)
from elspeth.engine.orchestrator.schema_reconstruction import (
    reconstruct_schema_from_json as reconstruct_schema_from_json,
)

_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")
_CSV_FORMULA_ESCAPE_PREFIX = "'"
_JSON_EXPORT_BATCH_SIZE = 1000
_PATH_TYPE = type(Path())


class _FlushableFile(Protocol):
    def flush(self) -> None: ...

    def fileno(self) -> int: ...

    def close(self) -> None: ...


class _FilesystemJsonlExportSink(Protocol):
    _path: Path
    _file: _FlushableFile | None

    def _claim_write_target(self) -> None: ...


def _neutralize_csv_formula_cell(value: Any) -> Any:
    """Prefix spreadsheet-formula-looking string cells for CSV audit exports."""
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
        return f"{_CSV_FORMULA_ESCAPE_PREFIX}{value}"
    return value


def _neutralize_csv_formula_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with spreadsheet-executable string cells neutralized."""
    return {key: _neutralize_csv_formula_cell(value) for key, value in record.items()}


class _CsvRecordTypeSpool:
    """Bounded-memory spool for one CSV record type."""

    def __init__(self) -> None:
        self._file = TemporaryFile("w+", newline="", encoding="utf-8")  # noqa: SIM115 - spool owns and closes this file
        self._writer = csv.writer(self._file)
        self.fieldnames: set[str] = set()
        self.count = 0

    def append(self, record: dict[str, Any]) -> None:
        self.fieldnames.update(record.keys())
        self._writer.writerow([len(record)])
        for key, value in record.items():
            self._writer.writerow([key, value])
        self.count += 1

    def iter_records(self) -> Iterator[dict[str, Any]]:
        self._file.seek(0)
        reader = csv.reader(self._file)
        while True:
            try:
                size_row = next(reader)
            except StopIteration:
                return
            if len(size_row) != 1:
                raise ValueError("CSV export spool is corrupt: record size row must have one column")
            field_count = int(size_row[0])
            record: dict[str, Any] = {}
            for _ in range(field_count):
                try:
                    key, value = next(reader)
                except StopIteration as exc:
                    raise ValueError("CSV export spool is corrupt: record ended early") from exc
                record[key] = value
            yield record

    def close(self) -> None:
        self._file.close()


class _FileSystemCsvAuditExportWriter:
    """Filesystem-backed writer for the multi-file CSV audit export capability."""

    def __init__(self, artifact_path: str) -> None:
        self._artifact_path = artifact_path

    @classmethod
    def from_sink(cls, *, sink_name: str, sink: SinkProtocol) -> _FileSystemCsvAuditExportWriter:
        artifact_path = sink.config.get("path")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError(f"CSV export requires file-based sink with 'path' in config, but sink '{sink_name}' has no path configured")
        return cls(artifact_path)

    def write(self, *, exporter: Any, run_id: str, sign: bool) -> None:
        _export_csv_multifile(
            exporter=exporter,
            run_id=run_id,
            artifact_path=self._artifact_path,
            sign=sign,
        )


def _write_json_export_batches(
    *,
    sink: SinkProtocol,
    ctx: PluginContext,
    records: Iterable[dict[str, Any]],
    batch_size: int = _JSON_EXPORT_BATCH_SIZE,
) -> tuple[int, int]:
    """Write JSON export records, batching only for sinks that publish incrementally."""
    if batch_size < 1:
        raise ValueError("JSON export batch size must be at least 1")

    if not _sink_supports_incremental_json_export_writes(sink):
        all_records = list(records)
        if not all_records:
            return 0, 0
        sink.write(all_records, ctx)
        return len(all_records), 1

    record_count = 0
    batches_written = 0
    batch: list[dict[str, Any]] = []

    with _jsonl_export_staging_target(sink):
        for record in records:
            batch.append(record)
            record_count += 1
            if len(batch) < batch_size:
                continue
            sink.write(batch, ctx)
            batches_written += 1
            batch = []

        if batch:
            sink.write(batch, ctx)
            batches_written += 1

    return record_count, batches_written


@contextmanager
def _jsonl_export_staging_target(sink: SinkProtocol) -> Iterator[None]:
    """Stage write-mode filesystem JSONL exports and publish only after full success."""
    final_path = _jsonl_filesystem_sink_path(sink)
    if final_path is None:
        yield
        return

    filesystem_sink = cast("_FilesystemJsonlExportSink", sink)
    filesystem_sink._claim_write_target()
    final_path = _jsonl_filesystem_sink_path(sink)
    if final_path is None:
        yield
        return

    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    filesystem_sink._path = temp_path
    try:
        yield
        _close_sink_file_if_open(sink)
        if temp_path.exists():
            os.replace(temp_path, final_path)
            _fsync_parent_directory(final_path)
    except BaseException:
        _close_sink_file_if_open(sink)
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        filesystem_sink._path = final_path


def _jsonl_filesystem_sink_path(sink: SinkProtocol) -> Path | None:
    if not _sink_supports_incremental_json_export_writes(sink):
        return None
    sink_attrs = vars(sink)
    if "_mode" in sink_attrs and sink_attrs["_mode"] == "append":
        return None
    if "_path" not in sink_attrs:
        return None
    path = sink_attrs["_path"]
    if type(path) is not _PATH_TYPE:
        return None
    return path


def _close_sink_file_if_open(sink: SinkProtocol) -> None:
    sink_attrs = vars(sink)
    if "_file" not in sink_attrs:
        return
    file_obj = sink_attrs["_file"]
    if file_obj is None:
        return
    open_file = cast("_FlushableFile", file_obj)
    open_file.flush()
    os.fsync(open_file.fileno())
    open_file.close()
    cast("_FilesystemJsonlExportSink", sink)._file = None


def _fsync_parent_directory(path: Path) -> None:
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _sink_supports_incremental_json_export_writes(sink: SinkProtocol) -> bool:
    """Return whether a JSON export sink can publish incrementally without rewriting cumulative content."""
    sink_attrs = vars(sink)
    if "_format" in sink_attrs:
        sink_format = sink_attrs["_format"]
        if type(sink_format) is not str:
            return False
        return sink_format == "jsonl"

    config = sink.config
    if type(config) is not dict:
        return False

    if "format" in config:
        configured_format = config["format"]
        if type(configured_format) is not str:
            return False
        return configured_format == "jsonl"

    if "path" not in config:
        return False
    path = config["path"]
    if type(path) is not str:
        return False
    return Path(path).suffix == ".jsonl"


def export_landscape(
    db: LandscapeDB,
    run_id: str,
    settings: ElspethSettings,
    sink_factory: Callable[[str], SinkProtocol],
) -> None:
    """Export audit trail to configured sink after run completion.

    For JSON format: writes all records to a single sink (records are
    heterogeneous but JSON handles that naturally).

    For CSV format: writes separate files per record_type to a directory,
    since CSV requires homogeneous schemas per file.

    Args:
        db: LandscapeDB instance for reading audit data
        run_id: The completed run ID
        settings: Full settings containing export configuration
        sink_factory: Creates a fresh, unstarted sink instance by name.
            The export path needs its own sink instance because the
            pipeline's sinks have already completed their lifecycle.

    Raises:
        ValueError: If signing requested but ELSPETH_SIGNING_KEY not set,
                   or if sink_factory raises for the configured sink name
    """
    from elspeth.core.landscape.exporter import LandscapeExporter

    export_config = settings.landscape.export

    # Get signing key from environment if signing enabled
    signing_key: bytes | None = None
    if export_config.sign:
        try:
            key_str = os.environ["ELSPETH_SIGNING_KEY"]
        except KeyError as exc:
            raise ValueError("ELSPETH_SIGNING_KEY environment variable required for signed export") from exc
        if not key_str.strip():
            raise ValueError("ELSPETH_SIGNING_KEY environment variable required for signed export")
        signing_key = key_str.encode("utf-8")

    # Create exporter
    exporter = LandscapeExporter(
        db,
        signing_key=signing_key,
        include_raw_error_rows=export_config.include_raw_error_rows,
    )

    sink_name = export_config.sink
    if sink_name is None:
        raise ValueError("Export sink name is None")
    sink = sink_factory(sink_name)
    sink.node_id = f"export:{sink_name}"

    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.landscape.factory import RecorderFactory

    factory = RecorderFactory(db)
    ctx = PluginContext(
        run_id=run_id, config={}, landscape=factory.plugin_audit_writer(), payload_store=factory.payload_store, node_id=sink.node_id
    )

    # Register the export sink as a node so the FK constraint in `operations` is satisfied.
    # The export sink writes post-run audit data and is not part of the execution graph,
    # so it must be registered here before begin_operation() is called.
    factory.data_flow.register_node(
        run_id=run_id,
        node_id=sink.node_id,
        plugin_name=sink.name,
        node_type=NodeType.SINK,
        plugin_version=sink.plugin_version,
        config=dict(sink.config),
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        determinism=Determinism.IO_WRITE,
        source_file_hash=sink.source_file_hash,
    )

    if export_config.format == "csv":
        csv_writer = _FileSystemCsvAuditExportWriter.from_sink(sink_name=sink_name, sink=sink)
        sink.on_start(ctx)
        try:
            with track_operation(
                factory.execution,
                run_id,
                sink.node_id,
                "sink_write",
                ctx,
                input_data={"export_format": "csv"},
            ):
                csv_writer.write(exporter=exporter, run_id=run_id, sign=export_config.sign)
                sink.flush()
        finally:
            sink.on_complete(ctx)
            sink.close()
    else:
        sink.on_start(ctx)
        try:
            with track_operation(
                factory.execution,
                run_id,
                sink.node_id,
                "sink_write",
                ctx,
                input_data={"export_format": "json"},
            ) as operation:
                record_count, batches_written = _write_json_export_batches(
                    sink=sink,
                    ctx=ctx,
                    records=exporter.export_run(run_id, sign=export_config.sign),
                )
                operation.output_data = {"record_count": record_count, "batches_written": batches_written}
                sink.flush()
        finally:
            sink.on_complete(ctx)
            sink.close()


def _export_csv_multifile(
    exporter: Any,  # LandscapeExporter (avoid circular import in type hint)
    run_id: str,
    artifact_path: str,
    sign: bool,
) -> None:
    """Export audit trail as multiple CSV files (one per record type).

    Creates a directory at the artifact path, then writes
    separate CSV files for each record type (run.csv, nodes.csv, etc.).

    Args:
        exporter: LandscapeExporter instance
        run_id: The completed run ID
        artifact_path: Path from sink config (validated by caller)
        sign: Whether to sign records
    """
    from elspeth.core.landscape.formatters import CSVFormatter

    export_dir = Path(artifact_path)
    if export_dir.suffix:
        # Remove file extension if present, treat as directory
        export_dir = export_dir.with_suffix("")

    export_dir.mkdir(parents=True, exist_ok=True)

    formatter = CSVFormatter()
    spools: dict[str, _CsvRecordTypeSpool] = {}

    try:
        for record_type, record in exporter.iter_run_records_by_type(run_id, sign=sign):
            spool = spools.get(record_type)
            if spool is None:
                spool = _CsvRecordTypeSpool()
                spools[record_type] = spool
            spool.append(_neutralize_csv_formula_record(formatter.format(record)))

        # Write each record type to its own CSV file.
        for record_type, spool in spools.items():
            if spool.count == 0:
                continue

            csv_path = export_dir / f"{record_type}.csv"
            fieldnames = sorted(spool.fieldnames)  # Sorted for determinism

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in spool.iter_records():
                    writer.writerow(rec)
    finally:
        for spool in spools.values():
            spool.close()
