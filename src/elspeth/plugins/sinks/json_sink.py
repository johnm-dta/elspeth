"""JSON sink plugin for ELSPETH.

Writes rows to JSON files. Supports JSON array and JSONL formats.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import hashlib
import io
import json
import os
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

from pydantic import model_validator

from elspeth.contracts import ArtifactDescriptor, PluginSchema
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.schema import SchemaConfig

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from elspeth.contracts.contexts import SinkContext
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import OutputCollisionPolicy, SinkPathConfig
from elspeth.plugins.infrastructure.display_headers import (
    apply_display_headers,
    get_effective_display_headers,
    init_display_headers,
    resolve_contract_from_context_if_needed,
    resolve_display_headers_if_needed,
    set_resume_field_resolution,
)
from elspeth.plugins.infrastructure.output_paths import (
    resolve_output_collision_path,
    should_create_exclusively,
    validate_output_collision_policy_mode,
)
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config


class JSONSinkConfig(SinkPathConfig):
    """Configuration for JSON sink plugin.

    Inherits from SinkPathConfig, which provides:
    - Path handling (from PathConfig)
    - Schema configuration (from DataPluginConfig)
    - Header output mode (headers: normalized | original | {mapping})
    """

    format: Literal["json", "jsonl"] | None = None
    indent: int | None = None
    encoding: str = "utf-8"
    mode: Literal["write", "append"] = "write"  # "write" (truncate) or "append"

    @model_validator(mode="after")
    def _validate_mode_format_compatibility(self) -> "JSONSinkConfig":
        """Reject append mode for JSON array format."""
        fmt = self.format
        if fmt is None:
            fmt = "jsonl" if Path(self.path).suffix == ".jsonl" else "json"

        if fmt == "json" and self.mode == "append":
            raise ValueError("JSONSink format='json' does not support mode='append'. Use format='jsonl' for append/resume output.")

        validate_output_collision_policy_mode(
            plugin_name="JSONSink",
            mode=self.mode,
            collision_policy=self.collision_policy,
        )
        return self


class JSONSink(BaseSink):
    """Write rows to a JSON file.

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Config options:
        path: Path to output JSON file (required)
        schema: Schema configuration (required, via PathConfig)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension.
        indent: Indentation for pretty-printing (default: None for compact)
        encoding: File encoding (default: "utf-8")

    The schema can be:
        - Observed: {"mode": "observed"} - accept any fields
        - Fixed: {"mode": "fixed", "fields": ["id: int", "name: str"]}
        - Flexible: {"mode": "flexible", "fields": ["id: int"]} - at least these fields
    """

    name = "json"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:cd491a54ae1a7a03"
    config_model = JSONSinkConfig
    # determinism inherited from BaseSink (IO_WRITE)

    # Note: supports_resume is set per-instance in __init__ based on format.
    # JSONL format supports resume (append), JSON array does not.
    # JSON array format rewrites the entire file on each write (seek(0) + truncate),
    # so it cannot append to existing output. JSONL writes line-by-line and can
    # append to existing files.
    _format: Literal["json", "jsonl"]
    _schema_config: SchemaConfig
    _collision_policy: OutputCollisionPolicy | None

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
        self._collision_policy = "append_or_create"

    def validate_output_target(self) -> "OutputValidationResult":
        """Validate existing JSONL file structure against configured schema.

        Reads the first line of the JSONL file to check field structure.

        Checks that:
        - Strict mode: Record fields match schema fields exactly (set comparison)
        - Free mode: All schema fields present (extras allowed)
        - Dynamic mode: No validation (schema adapts to existing structure)

        When display headers are configured (headers: original or headers: {mapping}),
        the existing file keys are display names, so we map expected schema fields
        to their display equivalents before comparison.

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
        if self._schema_config.is_observed:
            return OutputValidationResult.success(target_fields=existing)

        # Get expected fields from schema (guaranteed non-None when not dynamic)
        fields = self._schema_config.fields
        if fields is None:
            return OutputValidationResult.success(target_fields=existing)

        # Base expected fields are normalized schema names
        expected_normalized = [f.name for f in fields]

        # When display headers are configured, the file contains display names
        # Map expected fields to their display equivalents for comparison
        display_map = get_effective_display_headers(self)
        if display_map is not None:
            # Map normalized -> display for comparison against file keys
            expected = [display_map.get(f, f) for f in expected_normalized]
        else:
            expected = expected_normalized

        existing_set, expected_set = set(existing), set(expected)

        if self._schema_config.mode == "fixed":
            # Fixed: exact field match (set comparison)
            if existing_set != expected_set:
                return OutputValidationResult.failure(
                    message="JSONL record fields do not match schema (fixed mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(expected_set - existing_set),
                    extra_fields=sorted(existing_set - expected_set),
                )
        else:  # mode == "flexible"
            # Flexible: schema fields must exist (extras allowed)
            missing = expected_set - existing_set
            if missing:
                return OutputValidationResult.failure(
                    message="JSONL record missing required schema fields (flexible mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(missing),
                )

        return OutputValidationResult.success(target_fields=existing)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = JSONSinkConfig.from_dict(config, plugin_name=self.name)

        self._path = cfg.resolved_path()
        self._requested_path = self._path
        self._encoding = cfg.encoding
        self._indent = cfg.indent
        self._mode = cfg.mode
        self._collision_policy = cfg.collision_policy
        self._write_target_claimed = False
        if self._mode != "append":
            self._path = resolve_output_collision_path(self._requested_path, self._collision_policy)

        # Display header state (shared module handles all modes)
        init_display_headers(self, cfg.headers_mode, cfg.headers_mapping)

        # Auto-detect format from extension if not specified
        fmt = cfg.format
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

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

        # Required-field enforcement (centralized in SinkExecutor)
        self.declared_required_fields = self._schema_config.get_effective_required_fields()

        self._file: IO[str] | None = None
        self._rows: list[dict[str, Any]] = []  # Buffer for json array format

    def _claim_write_target(self) -> None:
        """Apply write-mode collision policy before the first filesystem mutation."""
        if self._write_target_claimed or self._mode == "append":
            return
        self._path = resolve_output_collision_path(self._requested_path, self._collision_policy)
        self._write_target_claimed = True

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        """Write a batch of rows to the JSON file.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes

        Raises:
            PluginContractViolation: Raised by executor if row fails input schema
                validation. This indicates a bug in an upstream transform.
        """
        if not rows:
            # Empty batch - return descriptor for empty content
            return SinkWriteResult(
                artifact=ArtifactDescriptor.for_file(
                    path=str(self._path),
                    content_hash=hashlib.sha256(b"").hexdigest(),
                    size_bytes=0,
                )
            )

        # Lazy resolution of contract from context for headers: original mode
        # ctx.contract is set by orchestrator after first valid source row
        # MUST happen BEFORE display header resolution (contract takes precedence over Landscape)
        resolve_contract_from_context_if_needed(self, ctx)

        # Lazy resolution of display headers from Landscape (fallback when no contract)
        # Must happen AFTER source iteration begins (when field resolution is recorded)
        resolve_display_headers_if_needed(self, ctx)

        # Apply display header mapping to row keys if configured
        output_rows = apply_display_headers(self, rows)

        if self._format == "jsonl":
            # Ensure file is open BEFORE capturing position — on first call
            # in append mode the file doesn't exist yet, so we must open it
            # first to get the correct position (end of existing content).
            self._ensure_jsonl_file_open()
            pre_write_pos = self._file.tell()  # type: ignore[union-attr]

            try:
                self._write_jsonl_content(output_rows)

                # Flush persistent file handle to ensure content is on disk for hashing.
                if self._file is not None:
                    self._file.flush()

                content_hash = self._compute_file_hash()
                size_bytes = self._path.stat().st_size
            except Exception:
                if self._file is not None and self._file.writable():
                    try:
                        self._file.seek(pre_write_pos)
                        self._file.truncate(pre_write_pos)
                        self._file.flush()
                        os.fsync(self._file.fileno())
                    except OSError as rollback_err:
                        raise RuntimeError(
                            f"JSONL write failed and rollback also failed — file may be corrupted at byte {pre_write_pos}"
                        ) from rollback_err
                raise
        else:
            # Buffer rows for JSON array format
            self._rows.extend(output_rows)
            # Write immediately (file is rewritten on each write for JSON format).
            # JSON array uses atomic temp-file writes (already safe), no rollback needed.
            self._write_json_array()

            content_hash = self._compute_file_hash()
            size_bytes = self._path.stat().st_size

        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=content_hash,
                size_bytes=size_bytes,
            )
        )

    def _ensure_jsonl_file_open(self) -> None:
        """Open the JSONL output file if not already open.

        Separated from content writing so the caller can capture file
        position between open and write — critical for correct rollback
        in append mode where pre-existing content must be preserved.
        """
        if self._file is None:
            if self._mode != "append":
                self._claim_write_target()
            file_mode = "a" if self._mode == "append" else "w"
            if self._mode != "append" and should_create_exclusively(self._collision_policy):
                file_mode = "x"

            # Validate schema compatibility before first append to existing file.
            # Without this, append mode can write rows with incompatible schemas
            # into the same JSONL file, violating sink schema contracts.
            if self._mode == "append" and self._path.exists() and not self._schema_config.is_observed:
                validation = self.validate_output_target()
                if not validation.valid:
                    msg_parts = [f"JSONL schema mismatch: {validation.error_message}"]
                    if validation.missing_fields:
                        msg_parts.append(f"Missing fields: {list(validation.missing_fields)}")
                    if validation.extra_fields:
                        msg_parts.append(f"Extra fields: {list(validation.extra_fields)}")
                    raise ValueError(". ".join(msg_parts))

            self._file = open(self._path, file_mode, encoding=self._encoding)  # noqa: SIM115

    def _write_jsonl_content(self, rows: list[dict[str, Any]]) -> None:
        """Stage and write JSONL rows to the already-open file handle."""
        if self._file is None:
            raise RuntimeError("JSONL file not open — call _ensure_jsonl_file_open() first")
        with io.StringIO() as staging:
            for row in rows:
                json.dump(row, staging, allow_nan=False)
                staging.write("\n")
            self._file.write(staging.getvalue())

    def _write_json_array(self) -> None:
        """Write buffered rows as JSON array (atomic write via temp file).

        Uses write-to-temp + fsync + os.replace() + dir fsync to prevent
        data loss on crash. The temp file is in the same directory to
        guarantee same-filesystem atomic rename on POSIX.

        On any failure, the temp file is cleaned up to prevent stale
        artifacts. The original file remains untouched until os.replace()
        succeeds.
        """
        if self._mode == "append":
            raise ValueError("JSONSink format='json' does not support mode='append'. Use format='jsonl' for append/resume output.")

        self._claim_write_target()
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            with open(temp_path, "w", encoding=self._encoding) as f:
                json.dump(self._rows, f, indent=self._indent, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            # Atomic replace — file transitions directly from old content to new
            os.replace(temp_path, self._path)
            # Fsync parent directory to ensure the rename is durable on power loss
            dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            # Clean up temp file on any failure (serialization, fsync, replace)
            # so stale .tmp files don't accumulate
            if temp_path.exists():
                temp_path.unlink()
            raise

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

        JSONL format: flushes the persistent file handle and fsyncs.
        JSON array format: no-op — _write_json_array already fsyncs
        via the atomic write pattern (temp file + fsync + os.replace).
        """
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the file handle and release buffered rows."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._rows = []

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        set_resume_field_resolution(self, resolution_mapping)
