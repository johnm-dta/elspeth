# src/elspeth/plugins/sinks/csv_sink.py
"""CSV sink plugin for ELSPETH.

Writes rows to CSV files with content hashing for audit integrity.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import csv
import hashlib
import os
from collections.abc import Sequence
from typing import IO, TYPE_CHECKING, Any, Literal

from elspeth.contracts import ArtifactDescriptor, PluginSchema

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class CSVSinkConfig(PathConfig):
    """Configuration for CSV sink plugin.

    Inherits from PathConfig, which requires schema configuration.
    """

    delimiter: str = ","
    encoding: str = "utf-8"
    validate_input: bool = False  # Optional runtime validation of incoming rows
    mode: Literal["write", "append"] = "write"


class CSVSink(BaseSink):
    """Write rows to a CSV file.

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Creates the CSV file on first write. When schema is explicit, headers are
    derived from schema field definitions. When schema is dynamic, headers are
    inferred from the first row's keys.

    Config options:
        path: Path to output CSV file (required)
        schema: Schema configuration (required, via PathConfig)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
        validate_input: Validate incoming rows against schema (default: False)
        mode: "write" (truncate, default) or "append" (add to existing file)

    The schema can be:
        - Dynamic: {"fields": "dynamic"} - accept any fields (headers inferred from first row)
        - Strict: {"mode": "strict", "fields": ["id: int", "name: str"]} - headers from schema
        - Free: {"mode": "free", "fields": ["id: int"]} - headers from schema, extras allowed

    Append mode behavior:
        - If file exists: reads headers from it and appends rows without header
        - If file doesn't exist or is empty: creates file with header (like write mode)
    """

    name = "csv"
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    # Resume capability: CSV can append to existing files
    supports_resume: bool = True

    def configure_for_resume(self) -> None:
        """Configure CSV sink for resume mode.

        Switches from truncate mode to append mode so resume operations
        add to existing output instead of overwriting.
        """
        self._mode = "append"

    def validate_output_target(self) -> "OutputValidationResult":
        """Validate existing CSV file headers against configured schema.

        Checks that:
        - Strict mode: Headers match schema fields exactly (including order)
        - Free mode: All schema fields present (extras allowed)
        - Dynamic mode: No validation (schema adapts to existing headers)

        Returns:
            OutputValidationResult indicating compatibility.
        """
        from elspeth.contracts.sink import OutputValidationResult

        # No file = valid (will create with correct headers)
        if not self._path.exists():
            return OutputValidationResult.success()

        # Read existing headers
        with open(self._path, encoding=self._encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self._delimiter)
            existing = list(reader.fieldnames or [])

        # Empty file = valid (will write headers on first write)
        if not existing:
            return OutputValidationResult.success()

        # Dynamic schema = no validation needed
        if self._schema_config.is_dynamic:
            return OutputValidationResult.success(target_fields=existing)

        # Get expected fields from schema (guaranteed non-None when not dynamic)
        fields = self._schema_config.fields
        if fields is None:
            return OutputValidationResult.success(target_fields=existing)
        expected = [f.name for f in fields]
        existing_set, expected_set = set(existing), set(expected)

        if self._schema_config.mode == "strict":
            # Strict: exact match including order
            if existing != expected:
                return OutputValidationResult.failure(
                    message="CSV headers do not match schema (strict mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(expected_set - existing_set),
                    extra_fields=sorted(existing_set - expected_set),
                    order_mismatch=(existing_set == expected_set),
                )
        else:  # mode == "free"
            # Free: schema fields must exist (extras allowed)
            missing = expected_set - existing_set
            if missing:
                return OutputValidationResult.failure(
                    message="CSV missing required schema fields (free mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(missing),
                )

        return OutputValidationResult.success(target_fields=existing)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSinkConfig.from_dict(config)

        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding
        self._validate_input = cfg.validate_input
        self._mode = cfg.mode

        # Store schema config for audit trail
        # PathConfig (via DataPluginConfig) ensures schema_config is not None
        self._schema_config = cfg.schema_config

        # CSV requires fixed-column structure - reject schemas that allow extra fields
        if self._schema_config.allows_extra_fields:
            raise ValueError(
                f"CSVSink requires fixed-column structure but schema allows_extra_fields=True "
                f"(mode={self._schema_config.mode!r}, is_dynamic={self._schema_config.is_dynamic}). "
                f"Use JSONSink for flexible schemas, or use mode='strict' for CSV output."
            )

        # CRITICAL: allow_coercion=False - wrong types are bugs, not data to fix
        # Sinks receive PIPELINE DATA (already validated by source)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "CSVRowSchema",
            allow_coercion=False,  # Sinks reject wrong types (upstream bug)
        )

        # Set input_schema for protocol compliance
        self.input_schema = self._schema_class

        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: Sequence[str] | None = None

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        """Write a batch of rows to the CSV file.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes

        Raises:
            ValidationError: If validate_input=True and a row fails validation.
                This indicates a bug in an upstream transform.
        """
        if not rows:
            # Empty batch - return descriptor for empty content
            return ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=hashlib.sha256(b"").hexdigest(),
                size_bytes=0,
            )

        # Optional input validation - crash on failure (upstream bug!)
        if self._validate_input and not self._schema_config.is_dynamic:
            for row in rows:
                # Raises ValidationError on failure - this is intentional
                self._schema_class.model_validate(row)

        # Lazy initialization - open file on first write
        if self._file is None:
            self._open_file(rows)

        # Write all rows in batch
        # Invariant: _file and _writer are always set together (by _open_file above)
        file = self._file
        writer = self._writer
        if file is None or writer is None:
            raise RuntimeError("CSVSink writer not initialized - this is a bug")
        for row in rows:
            writer.writerow(row)

        # Flush to ensure content is on disk for hashing
        file.flush()

        # Compute content hash from file
        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _open_file(self, rows: list[dict[str, Any]]) -> None:
        """Open file for writing, handling append mode.

        In append mode:
        - If file exists with headers: read headers from it, open in append mode
        - If file doesn't exist or is empty: create with headers (like write mode)

        In write mode:
        - Always truncate and write headers

        When schema is explicit (not dynamic), fieldnames are derived from schema
        field definitions. This ensures all defined fields (including optional ones)
        are present in the CSV header.

        When schema is dynamic, fieldnames are inferred from the first row's keys.

        Args:
            rows: First batch of rows (used to determine fieldnames if dynamic schema)
        """
        if self._mode == "append" and self._path.exists():
            # Try to read existing headers from file
            with open(self._path, encoding=self._encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=self._delimiter)
                existing_fieldnames = reader.fieldnames

            if existing_fieldnames:
                # Validate headers against explicit schema before opening
                # Dynamic schema = no validation (file headers are authoritative)
                if not self._schema_config.is_dynamic:
                    validation = self.validate_output_target()
                    if not validation.valid:
                        # Build clear error message
                        msg_parts = [f"CSV schema mismatch: {validation.error_message}"]
                        if validation.missing_fields:
                            msg_parts.append(f"Missing fields: {list(validation.missing_fields)}")
                        if validation.extra_fields:
                            msg_parts.append(f"Extra fields: {list(validation.extra_fields)}")
                        if validation.order_mismatch:
                            msg_parts.append("Fields present but in wrong order (strict mode)")
                        raise ValueError(". ".join(msg_parts))

                # Use existing headers, append mode (no header write)
                self._fieldnames = list(existing_fieldnames)
                self._file = open(  # noqa: SIM115 - handle kept open for streaming writes, closed in close()
                    self._path, "a", encoding=self._encoding, newline=""
                )
                self._writer = csv.DictWriter(
                    self._file,
                    fieldnames=self._fieldnames,
                    delimiter=self._delimiter,
                )
                # No header write - already exists
                return

        # Write mode OR append to non-existent/empty file
        # Determine fieldnames from schema (if explicit) or first row (if dynamic)
        self._fieldnames = self._get_fieldnames_from_schema_or_row(rows[0])
        self._file = open(  # noqa: SIM115 - handle kept open for streaming writes, closed in close()
            self._path, "w", encoding=self._encoding, newline=""
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            delimiter=self._delimiter,
        )
        self._writer.writeheader()

    def _get_fieldnames_from_schema_or_row(self, row: dict[str, Any]) -> list[str]:
        """Get fieldnames from schema or row keys.

        When schema is explicit, returns field names from schema definition.
        This ensures optional fields are present in the header.

        When schema is dynamic, falls back to inferring from row keys.
        """
        if not self._schema_config.is_dynamic and self._schema_config.fields:
            # Explicit schema: use field names from schema definition
            return [field_def.name for field_def in self._schema_config.fields]
        else:
            # Dynamic schema: infer from row keys
            return list(row.keys())

    def _compute_file_hash(self) -> str:
        """Compute SHA-256 hash of the file contents."""
        sha256 = hashlib.sha256()
        with open(self._path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def flush(self) -> None:
        """Flush buffered data to disk with fsync for durability.

        CRITICAL: Ensures data survives process crash and power loss.
        Called by orchestrator BEFORE creating checkpoints.

        This guarantees:
        - OS buffer flushed to disk (file.flush())
        - Filesystem metadata persisted (os.fsync())
        - Data durable on storage device
        """
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
