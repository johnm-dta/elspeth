# src/elspeth/engine/orchestrator/export.py
"""Post-run export and schema reconstruction functions.

This module handles:
1. Exporting the Landscape audit trail to JSON or CSV format after run completion
2. Reconstructing Pydantic schemas from JSON schema dictionaries (for pipeline resume)

The export functions support two modes:
- JSON format: All records written to a single sink (handles heterogeneous records)
- CSV format: Separate files per record type (CSV requires homogeneous schemas)

Schema reconstruction is needed when resuming a pipeline from checkpoint - the
original source schema is stored as JSON in the audit trail and must be
reconstructed to restore type fidelity (datetime, Decimal, etc.).

IMPORTANT: Import Cycle Prevention
----------------------------------
This module uses TYPE_CHECKING for imports that would cause cycles.
LandscapeDB is typed but imported conditionally to avoid circular imports.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings
    from elspeth.core.landscape import LandscapeDB
    from elspeth.plugins.protocols import SinkProtocol

from elspeth.contracts.plugin_context import PluginContext


def export_landscape(
    db: LandscapeDB,
    run_id: str,
    settings: ElspethSettings,
    sinks: dict[str, SinkProtocol],
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
        sinks: Dict of sink_name -> sink instance from PipelineConfig

    Raises:
        ValueError: If signing requested but ELSPETH_SIGNING_KEY not set,
                   or if configured sink not found
    """
    from elspeth.core.landscape.exporter import LandscapeExporter

    export_config = settings.landscape.export

    # Get signing key from environment if signing enabled
    signing_key: bytes | None = None
    if export_config.sign:
        try:
            key_str = os.environ["ELSPETH_SIGNING_KEY"]
        except KeyError:
            raise ValueError("ELSPETH_SIGNING_KEY environment variable required for signed export") from None
        signing_key = key_str.encode("utf-8")

    # Create exporter
    exporter = LandscapeExporter(db, signing_key=signing_key)

    # Get target sink config
    sink_name = export_config.sink
    if sink_name not in sinks:
        raise ValueError(f"Export sink '{sink_name}' not found in sinks")
    sink = sinks[sink_name]

    # Create context for sink writes.
    # LandscapeRecorder is needed by sinks that restore source headers
    # (restore_source_headers=True calls ctx.landscape.get_source_field_resolution).
    # Import inside function body to avoid circular imports (see module docstring).
    from elspeth.core.landscape.recorder import LandscapeRecorder

    recorder = LandscapeRecorder(db)
    ctx = PluginContext(run_id=run_id, config={}, landscape=recorder)

    if export_config.format == "csv":
        # Multi-file CSV export: one file per record type
        # CSV export writes files directly (not via sink.write), so we need
        # the path from sink config. CSV format requires file-based sink.
        if "path" not in sink.config:
            raise ValueError(f"CSV export requires file-based sink with 'path' in config, but sink '{sink_name}' has no path configured")
        artifact_path: str = sink.config["path"]
        _export_csv_multifile(
            exporter=exporter,
            run_id=run_id,
            artifact_path=artifact_path,
            sign=export_config.sign,
            ctx=ctx,
        )
    else:
        # JSON export: batch all records for single write
        records = list(exporter.export_run(run_id, sign=export_config.sign))
        try:
            if records:
                # Capture ArtifactDescriptor for audit trail (future use)
                _artifact_descriptor = sink.write(records, ctx)
            sink.flush()
        finally:
            sink.close()


def _export_csv_multifile(
    exporter: Any,  # LandscapeExporter (avoid circular import in type hint)
    run_id: str,
    artifact_path: str,
    sign: bool,
    ctx: PluginContext,  # - reserved for future use
) -> None:
    """Export audit trail as multiple CSV files (one per record type).

    Creates a directory at the artifact path, then writes
    separate CSV files for each record type (run.csv, nodes.csv, etc.).

    Args:
        exporter: LandscapeExporter instance
        run_id: The completed run ID
        artifact_path: Path from sink config (validated by caller)
        sign: Whether to sign records
        ctx: Plugin context for sink operations (reserved for future use)
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
        flat_records = [formatter.format(r) for r in records]

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


def reconstruct_schema_from_json(schema_dict: Mapping[str, object]) -> type:
    """Reconstruct Pydantic schema class from JSON schema dict.

    Handles complete Pydantic JSON schema including:
    - Primitive types: string, integer, number, boolean
    - datetime: string with format="date-time"
    - Decimal: anyOf with number/string (for precision preservation)
    - Arrays: type="array" with items schema
    - Nested objects: type="object" with properties schema

    Args:
        schema_dict: Pydantic JSON schema dict (from model_json_schema())

    Returns:
        Dynamically created Pydantic model class

    Raises:
        ValueError: If schema is malformed, empty, or contains unsupported types
    """
    from pydantic import ConfigDict, create_model

    from elspeth.contracts import PluginSchema

    # Extract field definitions from Pydantic JSON schema
    # This is OUR data (from Landscape DB) - crash if malformed
    if "properties" not in schema_dict:
        raise ValueError(
            "Resume failed: Schema JSON has no 'properties' field. This indicates a malformed schema. Cannot reconstruct types."
        )
    properties = cast(Mapping[str, object], schema_dict["properties"])

    # Handle observed/dynamic schemas: empty properties with additionalProperties=true
    # This is the normal JSON schema output for schema.mode=observed (dynamic schemas)
    # See schema_factory._create_dynamic_schema for the creation side
    if not properties:
        if "additionalProperties" in schema_dict and schema_dict["additionalProperties"] is True:
            # Dynamic schema - accepts any fields, no fixed properties
            return create_model(
                "RestoredDynamicSchema",
                __base__=PluginSchema,
                __config__=ConfigDict(extra="allow"),
            )
        # Empty properties WITHOUT additionalProperties=true is genuinely malformed
        raise ValueError(
            "Resume failed: Schema has zero fields defined and additionalProperties is not true. "
            "Cannot resume with empty fixed schema - this would silently discard all row data. "
            "For dynamic schemas, additionalProperties must be true."
        )

    # "required" is optional in JSON Schema spec - empty list is valid default
    required_fields = set(cast(list[str], schema_dict["required"])) if "required" in schema_dict else set()

    # Resolve top-level fields recursively, preserving array item types and
    # nested object property schemas.
    return _create_schema_model(
        model_name="RestoredSourceSchema",
        properties=properties,
        required_fields=required_fields,
        schema_defs=cast(Mapping[str, object], schema_dict["$defs"]) if "$defs" in schema_dict else None,
        create_model=create_model,
        schema_base=PluginSchema,
    )


def _create_schema_model(
    model_name: str,
    properties: Mapping[str, object],
    required_fields: set[str],
    *,
    schema_defs: Mapping[str, object] | None,
    create_model: Any,
    schema_base: Any,
) -> type:
    """Create a Pydantic model from JSON schema properties."""
    field_definitions: dict[str, Any] = {}

    for field_name, raw_field_info in properties.items():
        field_info = cast(Mapping[str, object], raw_field_info)
        field_type = _json_schema_to_python_type(
            field_name,
            field_info,
            schema_defs=schema_defs,
            create_model=create_model,
            schema_base=schema_base,
        )
        field_definitions[field_name] = (field_type, ... if field_name in required_fields else None)

    return cast(type, create_model(model_name, __base__=schema_base, **field_definitions))


def _model_name_for_field(field_name: str) -> str:
    """Build a deterministic nested model name from a field name."""
    tokens = re.findall(r"[A-Za-z0-9]+", field_name)
    if not tokens:
        return "RestoredNestedSchema"
    title_cased = "".join(token[:1].upper() + token[1:] for token in tokens)
    if title_cased[0].isdigit():
        title_cased = f"Field{title_cased}"
    return f"Restored{title_cased}Schema"


def _json_schema_to_python_type(
    field_name: str,
    field_info: Mapping[str, object],
    *,
    schema_defs: Mapping[str, object] | None = None,
    create_model: Any | None = None,
    schema_base: Any | None = None,
) -> Any:
    """Map Pydantic JSON schema field to Python type.

    Handles Pydantic's type mapping including special cases:
    - datetime: {"type": "string", "format": "date-time"}
    - date: {"type": "string", "format": "date"}
    - time: {"type": "string", "format": "time"}
    - timedelta: {"type": "string", "format": "duration"}
    - UUID: {"type": "string", "format": "uuid"}
    - Decimal: {"anyOf": [{"type": "number"}, {"type": "string"}]}
    - Nullable: {"anyOf": [{"type": "T"}, {"type": "null"}]} -> T
    - Nullable ref: {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}]} -> M
    - list[T]: {"type": "array", "items": {...}}
    - dict: {"type": "object"} without properties

    Args:
        field_name: Field name (for error messages)
        field_info: JSON schema field definition

    Returns:
        Python type for Pydantic field

    Raises:
        ValueError: If field type is not supported (prevents silent degradation)
    """
    from datetime import date, datetime, time, timedelta
    from decimal import Decimal
    from uuid import UUID

    # Handle anyOf patterns FIRST (before checking for "type" key)
    # anyOf is used for: Decimal, nullable types (T | None)
    if "anyOf" in field_info:
        any_of_items = cast(list[Mapping[str, object]], field_info["anyOf"])

        # Pattern 1: Decimal - {"anyOf": [{"type": "number"}, {"type": "string", ...}]}
        type_strs = {cast(str, item["type"]) for item in any_of_items if "type" in item}
        if {"number", "string"}.issubset(type_strs) and "null" not in type_strs:
            return Decimal

        # Pattern 2: Nullable - {"anyOf": [{"type": "T", ...}, {"type": "null"}]}
        #   or with $ref:  {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}]}
        # Extract the non-null type and recursively resolve it
        if "null" in type_strs:
            # Items without "type" key (e.g. $ref entries) are non-null by definition
            non_null_items = [item for item in any_of_items if item.get("type") != "null"]
            if len(non_null_items) == 1:
                # Recursively resolve the non-null type, then wrap as Optional.
                # Returning T | None (not bare T) is critical: Pydantic model types
                # reject None unless the type annotation explicitly includes it.
                inner_type = _json_schema_to_python_type(
                    field_name,
                    non_null_items[0],
                    schema_defs=schema_defs,
                    create_model=create_model,
                    schema_base=schema_base,
                )
                return inner_type | None

        # Unsupported anyOf pattern (e.g., Union[str, int] without null)
        raise ValueError(
            f"Resume failed: Field '{field_name}' has unsupported anyOf pattern. "
            f"Supported patterns: Decimal (number|string), nullable (T|null). "
            f"Schema definition: {field_info}. "
            f"This is a bug in schema reconstruction - please report this."
        )

    # Resolve local references in Pydantic schemas (e.g., "#/$defs/NestedModel")
    if "$ref" in field_info:
        ref = cast(str, field_info["$ref"])
        ref_prefix = "#/$defs/"
        if not ref.startswith(ref_prefix):
            raise ValueError(
                f"Resume failed: Field '{field_name}' has unsupported $ref '{ref}'. Only local refs under '#/$defs/' are supported."
            )
        if schema_defs is None:
            raise ValueError(f"Resume failed: Field '{field_name}' references '{ref}' but schema has no $defs section.")
        def_name = ref[len(ref_prefix) :]
        if def_name not in schema_defs:
            raise ValueError(f"Resume failed: Field '{field_name}' references missing schema def '{def_name}'.")
        return _json_schema_to_python_type(
            field_name,
            cast(Mapping[str, object], schema_defs[def_name]),
            schema_defs=schema_defs,
            create_model=create_model,
            schema_base=schema_base,
        )

    # Get basic type - required for all non-anyOf fields
    if "type" not in field_info:
        raise ValueError(
            f"Resume failed: Field '{field_name}' has no 'type' in schema. "
            f"Schema definition: {field_info}. "
            f"Cannot determine Python type for field."
        )
    field_type_str = cast(str, field_info["type"])

    # Handle string types with format specifiers
    if field_type_str == "string":
        fmt = None
        if "format" in field_info:
            fmt = field_info["format"]
        if fmt == "date-time":
            return datetime
        if fmt == "date":
            return date
        if fmt == "time":
            return time
        if fmt == "duration":
            return timedelta
        if fmt == "uuid":
            return UUID
        # Plain string (no format or unknown format)
        return str

    # Handle array types
    if field_type_str == "array":
        # "items" is optional in JSON Schema arrays. When present, recursively
        # restore item type fidelity (e.g., list[int], list[NestedSchema]).
        if "items" in field_info:
            item_info = cast(Mapping[str, object], field_info["items"])
            item_type = _json_schema_to_python_type(
                f"{field_name}_item",
                item_info,
                schema_defs=schema_defs,
                create_model=create_model,
                schema_base=schema_base,
            )
            return list.__class_getitem__(item_type)
        return list

    # Handle nested object types
    if field_type_str == "object":
        # Typed nested object: recursively create a nested schema model.
        if "properties" in field_info:
            properties = cast(Mapping[str, object], field_info["properties"])
        else:
            properties = None
        if properties:
            nested_required = set(cast(list[str], field_info["required"])) if "required" in field_info else set()
            nested_name = _model_name_for_field(field_name)
            return _create_schema_model(
                model_name=nested_name,
                properties=properties,
                required_fields=nested_required,
                schema_defs=schema_defs,
                create_model=create_model,
                schema_base=schema_base,
            )

        # Map additionalProperties schemas when present (e.g., dict[str, int]).
        if "additionalProperties" in field_info:
            additional = field_info["additionalProperties"]
            if additional is True:
                return dict[str, Any]
            if type(additional) is dict:
                value_type = _json_schema_to_python_type(
                    f"{field_name}_value",
                    cast(Mapping[str, object], additional),
                    schema_defs=schema_defs,
                    create_model=create_model,
                    schema_base=schema_base,
                )
                return dict.__class_getitem__((str, value_type))
            if additional is False:
                return dict
            raise ValueError(f"Resume failed: Field '{field_name}' has invalid additionalProperties value: {additional!r}.")

        # Generic dict (no specific structure)
        return dict

    # Handle other primitive types
    primitive_type_map = {
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    if field_type_str in primitive_type_map:
        return primitive_type_map[field_type_str]

    # Unknown type - CRASH instead of silent degradation
    raise ValueError(
        f"Resume failed: Field '{field_name}' has unsupported type '{field_type_str}'. "
        f"Supported types: string, integer, number, boolean, date-time, date, time, "
        f"duration, uuid, Decimal, array, object. "
        f"Schema definition: {field_info}. "
        f"This is a bug in schema reconstruction - please report this."
    )
