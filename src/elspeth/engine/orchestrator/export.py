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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _neutralize_csv_formula_cell(value: Any) -> Any:
    """Prefix spreadsheet-formula-looking string cells for CSV audit exports."""
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
        return f"{_CSV_FORMULA_ESCAPE_PREFIX}{value}"
    return value


def _neutralize_csv_formula_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with spreadsheet-executable string cells neutralized."""
    return {key: _neutralize_csv_formula_cell(value) for key, value in record.items()}


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
        records = list(exporter.export_run(run_id, sign=export_config.sign))
        sink.on_start(ctx)
        try:
            with track_operation(
                factory.execution,
                run_id,
                sink.node_id,
                "sink_write",
                ctx,
                input_data={"export_format": "json", "record_count": len(records)},
            ):
                if records:
                    sink.write(records, ctx)
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

    # Get records grouped by type
    grouped = exporter.export_run_grouped(run_id, sign=sign)
    formatter = CSVFormatter()

    # Write each record type to its own CSV file
    for record_type, records in grouped.items():
        if not records:
            continue

        csv_path = export_dir / f"{record_type}.csv"

        # Flatten all records for CSV
        flat_records = [_neutralize_csv_formula_record(formatter.format(r)) for r in records]

        # Get union of all keys (some records may have optional fields)
        all_keys: set[str] = set()
        for rec in flat_records:
            all_keys.update(rec.keys())
        fieldnames = sorted(all_keys)  # Sorted for determinism

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in flat_records:
                writer.writerow(rec)
