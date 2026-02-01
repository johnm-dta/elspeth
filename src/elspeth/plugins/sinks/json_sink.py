# src/elspeth/plugins/sinks/json_sink.py
"""JSON sink plugin for ELSPETH.

Writes rows to JSON files. Supports JSON array and JSONL formats.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import hashlib
import json
import os
from typing import IO, TYPE_CHECKING, Any, Literal

from elspeth.contracts import ArtifactDescriptor, PluginSchema

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class JSONSinkConfig(PathConfig):
    """Configuration for JSON sink plugin.

    Inherits from PathConfig, which requires schema configuration.
    """

    format: Literal["json", "jsonl"] | None = None
    indent: int | None = None
    encoding: str = "utf-8"
    validate_input: bool = False  # Optional runtime validation of incoming rows
    mode: Literal["write", "append"] = "write"  # "write" (truncate) or "append"


class JSONSink(BaseSink):
    """Write rows to a JSON file.

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Config options:
        path: Path to output JSON file (required)
        schema: Schema configuration (required, via PathConfig)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension.
        indent: Indentation for pretty-printing (default: None for compact)
        encoding: File encoding (default: "utf-8")
        validate_input: Validate incoming rows against schema (default: False)

    The schema can be:
        - Dynamic: {"fields": "dynamic"} - accept any fields
        - Strict: {"mode": "strict", "fields": ["id: int", "name: str"]}
        - Free: {"mode": "free", "fields": ["id: int"]} - at least these fields
    """

    name = "json"
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    # Note: supports_resume is set per-instance in __init__ based on format.
    # JSONL format supports resume (append), JSON array does not.
    # JSON array format rewrites the entire file on each write (seek(0) + truncate),
    # so it cannot append to existing output. JSONL writes line-by-line and can
    # append to existing files.

    def configure_for_resume(self) -> None:
        """Configure JSON sink for resume mode.

        Only JSONL format supports resume. JSON array format rewrites the
        entire file on each write and cannot append.

        Raises:
            NotImplementedError: If format is JSON array (not JSONL).
        """
        if self._format != "jsonl":
            raise NotImplementedError(
                f"JSONSink with format='{self._format}' does not support resume. "
                f"JSON array format rewrites the entire file and cannot append. "
                f"Use format='jsonl' for resumable JSON output."
            )
        self._mode = "append"

    def validate_output_target(self) -> "OutputValidationResult":
        """Validate existing JSONL file structure against configured schema.

        Reads the first line of the JSONL file to check field structure.

        Checks that:
        - Strict mode: Record fields match schema fields exactly (set comparison)
        - Free mode: All schema fields present (extras allowed)
        - Dynamic mode: No validation (schema adapts to existing structure)

        Note: Only JSONL format supports resume. JSON array returns valid=True
        (it will overwrite anyway).

        Returns:
            OutputValidationResult indicating compatibility.
        """
        from elspeth.contracts.sink import OutputValidationResult

        # Only JSONL supports resume - JSON array rewrites entirely
        if self._format != "jsonl":
            return OutputValidationResult.success()

        # No file or empty file = valid (will create on first write)
        if not self._path.exists() or self._path.stat().st_size == 0:
            return OutputValidationResult.success()

        # Read first line to check structure
        with open(self._path, encoding=self._encoding) as f:
            first_line = f.readline().strip()
            if not first_line:
                return OutputValidationResult.success()
            try:
                first_record = json.loads(first_line)
            except json.JSONDecodeError:
                return OutputValidationResult.failure(message="Existing JSONL file contains invalid JSON")

            # Ensure first record is a dict (JSONL should contain objects)
            if not isinstance(first_record, dict):
                return OutputValidationResult.failure(message="Existing JSONL file contains non-object records")

            existing = list(first_record.keys())

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
            # Strict: exact field match (set comparison)
            if existing_set != expected_set:
                return OutputValidationResult.failure(
                    message="JSONL record fields do not match schema (strict mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(expected_set - existing_set),
                    extra_fields=sorted(existing_set - expected_set),
                )
        else:  # mode == "free"
            # Free: schema fields must exist (extras allowed)
            missing = expected_set - existing_set
            if missing:
                return OutputValidationResult.failure(
                    message="JSONL record missing required schema fields (free mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(missing),
                )

        return OutputValidationResult.success(target_fields=existing)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = JSONSinkConfig.from_dict(config)

        self._path = cfg.resolved_path()
        self._encoding = cfg.encoding
        self._indent = cfg.indent
        self._validate_input = cfg.validate_input

        # Auto-detect format from extension if not specified
        fmt = cfg.format
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt
        self._mode = cfg.mode

        # Set resume capability based on format
        # JSONL can append; JSON array rewrites entirely and cannot resume
        self.supports_resume = fmt == "jsonl"

        # Store schema config for audit trail
        # PathConfig (via DataPluginConfig) ensures schema_config is not None
        self._schema_config = cfg.schema_config

        # CRITICAL: allow_coercion=False - wrong types are bugs, not data to fix
        # Sinks receive PIPELINE DATA (already validated by source)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "JSONRowSchema",
            allow_coercion=False,  # Sinks reject wrong types (upstream bug)
        )

        # Set input_schema for protocol compliance
        self.input_schema = self._schema_class

        self._file: IO[str] | None = None
        self._rows: list[dict[str, Any]] = []  # Buffer for json array format

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        """Write a batch of rows to the JSON file.

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

        if self._format == "jsonl":
            self._write_jsonl_batch(rows)
        else:
            # Buffer rows for JSON array format
            self._rows.extend(rows)
            # Write immediately (file is rewritten on each write for JSON format)
            self._write_json_array()

        # Flush to ensure content is on disk for hashing
        if self._file is not None:
            self._file.flush()

        # Compute content hash from file
        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _write_jsonl_batch(self, rows: list[dict[str, Any]]) -> None:
        """Write rows as JSONL.

        Uses write mode (truncate) or append mode based on self._mode.
        Append mode is used during resume to add to existing output.
        """
        if self._file is None:
            file_mode = "a" if self._mode == "append" else "w"
            self._file = open(self._path, file_mode, encoding=self._encoding)  # noqa: SIM115

        for row in rows:
            json.dump(row, self._file)
            self._file.write("\n")

    def _write_json_array(self) -> None:
        """Write buffered rows as JSON array (rewrite mode)."""
        if self._file is None:
            # Handle kept open for streaming writes, closed in close()
            self._file = open(self._path, "w", encoding=self._encoding)  # noqa: SIM115
        self._file.seek(0)
        self._file.truncate()
        json.dump(self._rows, self._file, indent=self._indent)

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
            self._rows = []

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
