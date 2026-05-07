"""CSV sink plugin for ELSPETH.

Writes rows to CSV files with content hashing for audit integrity.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from collections.abc import Sequence
from typing import IO, TYPE_CHECKING, Any, Literal

from pydantic import field_validator, model_validator

from elspeth.contracts import ArtifactDescriptor, PluginSchema
from elspeth.contracts.diversion import SinkWriteResult

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from elspeth.contracts.contexts import SinkContext
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import OutputCollisionPolicy, SinkPathConfig
from elspeth.plugins.infrastructure.display_headers import (
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


class CSVSinkConfig(SinkPathConfig):
    """Configuration for CSV sink plugin.

    Inherits from SinkPathConfig, which provides:
    - Path handling (from PathConfig)
    - Schema configuration (from DataPluginConfig)
    - Header output mode (headers: normalized | original | {mapping})
    """

    delimiter: str = ","
    encoding: str = "utf-8"
    mode: Literal["write", "append"] = "write"

    @field_validator("delimiter")
    @classmethod
    def _validate_delimiter(cls, v: str) -> str:
        if len(v) != 1:
            raise ValueError(f"delimiter must be a single character, got {v!r}")
        return v

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, v: str) -> str:
        import codecs

        try:
            codecs.lookup(v)
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {v!r}") from exc
        return v

    @model_validator(mode="after")
    def _validate_collision_policy_mode(self) -> CSVSinkConfig:
        validate_output_collision_policy_mode(
            plugin_name="CSVSink",
            mode=self.mode,
            collision_policy=self.collision_policy,
        )
        return self


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
        mode: "write" (truncate, default) or "append" (add to existing file)

    The schema can be (all use infer-and-lock pattern):
        - Fixed: {"mode": "fixed", "fields": [...]} - columns from config, extras rejected
        - Flexible: {"mode": "flexible", "fields": [...]} - declared + first-row extras, then locked
        - Observed: {"mode": "observed"} - columns from first row, then locked

    Append mode behavior:
        - If file exists: reads headers from it and appends rows without header
        - If file doesn't exist or is empty: creates file with header (like write mode)
    """

    name = "csv"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:c24ad00a6582427b"
    config_model = CSVSinkConfig
    # determinism inherited from BaseSink (IO_WRITE)

    # Resume capability: CSV can append to existing files
    supports_resume: bool = True
    _collision_policy: OutputCollisionPolicy | None

    def configure_for_resume(self) -> None:
        """Configure CSV sink for resume mode.

        Switches from truncate mode to append mode so resume operations
        add to existing output instead of overwriting.
        """
        self._mode = "append"
        self._collision_policy = "append_or_create"

    def validate_output_target(self) -> OutputValidationResult:
        """Validate existing CSV file headers against configured schema.

        Checks that:
        - Strict mode: Headers match schema fields exactly (including order)
        - Free mode: All schema fields present (extras allowed)
        - Dynamic mode: No validation (schema adapts to existing headers)

        When display headers are configured (headers: original or headers: {mapping}),
        the existing file headers are display names, so we map expected schema fields
        to their display equivalents before comparison.

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
        from elspeth.contracts.header_modes import HeaderMode

        display_map = get_effective_display_headers(self)
        if display_map is not None:
            # Map normalized -> display for comparison against file headers
            expected = [display_map.get(f, f) for f in expected_normalized]
        elif self._headers_mode == HeaderMode.ORIGINAL:
            # ORIGINAL mode but field_resolution not yet available — comparing
            # normalized names against display-name file headers would be wrong.
            # Skip validation; it will be checked once headers are resolved.
            return OutputValidationResult.success(target_fields=existing)
        else:
            expected = expected_normalized

        existing_set, expected_set = set(existing), set(expected)

        if self._schema_config.mode == "fixed":
            # Fixed: exact match including order
            if existing != expected:
                return OutputValidationResult.failure(
                    message="CSV headers do not match schema (fixed mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(expected_set - existing_set),
                    extra_fields=sorted(existing_set - expected_set),
                    order_mismatch=(existing_set == expected_set),
                )
        else:  # mode == "flexible"
            # Flexible: schema fields must exist (extras allowed)
            missing = expected_set - existing_set
            if missing:
                return OutputValidationResult.failure(
                    message="CSV missing required schema fields (flexible mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(missing),
                )

        return OutputValidationResult.success(target_fields=existing)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSinkConfig.from_dict(config, plugin_name=self.name)

        self._path = cfg.resolved_path()
        self._requested_path = self._path
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding
        self._mode = cfg.mode
        self._collision_policy = cfg.collision_policy
        self._write_target_claimed = False
        if self._mode != "append":
            self._path = resolve_output_collision_path(self._requested_path, self._collision_policy)

        # Display header state (shared module handles all modes)
        init_display_headers(self, cfg.headers_mode, cfg.headers_mapping)

        # Store schema config for audit trail
        # PathConfig (via DataPluginConfig) ensures schema_config is not None
        self._schema_config = cfg.schema_config

        # CSV supports all schema modes via infer-and-lock:
        # - mode='fixed': columns from config, extras rejected at write time
        # - mode='flexible': declared columns + extras from first row, then locked
        # - mode='observed': columns from first row, then locked
        #
        # DictWriter's default extrasaction='raise' enforces the lock - any row
        # with fields not in the established fieldnames will error.

        # CRITICAL: allow_coercion=False - wrong types are bugs, not data to fix
        # Sinks receive PIPELINE DATA (already validated by source)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "CSVRowSchema",
            allow_coercion=False,  # Sinks reject wrong types (upstream bug)
        )

        # Set input_schema for protocol compliance
        self.input_schema = self._schema_class

        # Required-field enforcement (centralized in SinkExecutor)
        self.declared_required_fields = self._schema_config.get_effective_required_fields()

        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: Sequence[str] | None = None
        # Incremental hasher — avoids O(N²) full-file re-reads in append mode
        self._hasher: hashlib._Hash | None = None

    def _claim_write_target(self) -> None:
        """Apply write-mode collision policy before the first filesystem mutation."""
        if self._write_target_claimed or self._mode == "append":
            return
        self._path = resolve_output_collision_path(self._requested_path, self._collision_policy)
        self._write_target_claimed = True

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        """Write a batch of rows to the CSV file.

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

        # First-call initialization: compute fieldnames and stage BEFORE
        # opening the file in write mode.  In write mode, _open_file truncates
        # the file and writes headers.  If we did that first and staging then
        # failed (e.g. DictWriter rejects a row with extra/missing fields),
        # the file would already be truncated — prior contents lost.
        #
        # For append mode the file already exists so truncation is not a
        # concern; we follow the original open-then-stage order.
        first_call = self._file is None

        if first_call and self._mode != "append":
            # Write mode, first batch: compute fieldnames from schema/row
            # without touching the filesystem.
            data_fields, display_fields = self._get_field_names_and_display(rows[0])
            self._fieldnames = data_fields

            # Stage rows in memory — if any row is invalid, we fail here
            # BEFORE the file is created/truncated.
            staging_buffer = io.StringIO()
            staging_writer = csv.DictWriter(
                staging_buffer,
                fieldnames=data_fields,
                delimiter=self._delimiter,
            )
            for row in rows:
                staging_writer.writerow(row)
            staged_content = staging_buffer.getvalue()

            # Staging succeeded — now safe to open (truncate) and write.
            self._open_file_write_mode(data_fields, display_fields)

            # Write staged rows after the header
            file = self._file
            if file is None or self._hasher is None:
                raise RuntimeError("CSVSink writer not initialized - this is a bug")

            # Mutation phase: write, flush, hash, stat.
            # If anything fails after the file was created by _open_file_write_mode,
            # clean up the newly-created file entirely — it has no prior content
            # to preserve, and leaving a partial file violates audit integrity.
            try:
                pre_write_pos = file.tell()
                file.write(staged_content)
                file.flush()

                # Incremental hash: only read newly written bytes (O(batch) not O(file))
                with open(self._path, "rb") as bf:
                    bf.seek(pre_write_pos)
                    for chunk in iter(lambda: bf.read(8192), b""):
                        self._hasher.update(chunk)
                content_hash = self._hasher.hexdigest()
                size_bytes = self._path.stat().st_size
            except Exception:
                # Write-mode first batch: file was just created by us — no
                # pre-existing content to preserve.  Remove it entirely so a
                # partial file doesn't masquerade as a valid artifact.  The
                # re-raised exception carries all diagnostic context.
                file.close()
                self._file = None
                self._writer = None
                self._hasher = None
                if self._path.exists():
                    self._path.unlink()
                raise
        else:
            # Append mode first call, or any subsequent call
            if first_call:
                self._open_file(rows)

            # Write all rows in batch
            # Invariant: _file and _writer are always set together (by _open_file above)
            file = self._file
            writer = self._writer
            if file is None or writer is None:
                raise RuntimeError("CSVSink writer not initialized - this is a bug")
            if self._hasher is None:
                raise RuntimeError("CSVSink hasher not initialized - this is a bug")

            # Stage the entire batch in memory BEFORE writing to file.
            # This prevents partial writes: if any row fails serialization (e.g.,
            # extra fields rejected by DictWriter), NO rows are written to disk.
            # Without this, row N failing after rows 0..N-1 are written causes
            # audit divergence -- CSV has rows the Landscape marks as FAILED.
            staging_buffer = io.StringIO()
            fieldnames = self._fieldnames
            if fieldnames is None:
                raise RuntimeError("write() called before _fieldnames set by _write_header()")
            staging_writer = csv.DictWriter(
                staging_buffer,
                fieldnames=fieldnames,
                delimiter=self._delimiter,
            )
            for row in rows:
                staging_writer.writerow(row)
            staged_content = staging_buffer.getvalue()

            # Track write position before writing for incremental hashing.
            # Use file.tell() not stat().st_size — stat() has a TOCTOU race where
            # another process could append between stat and write, causing the hasher
            # to incorporate foreign bytes.
            pre_write_pos = file.tell()

            # Mutation phase: write, flush, hash, stat.
            # If anything after write() fails, the file has new bytes with no
            # corresponding audit record.  Rollback by truncating to pre_write_pos.
            try:
                file.write(staged_content)
                file.flush()

                # Incremental hash: only read newly written bytes (O(batch) not O(file))
                with open(self._path, "rb") as bf:
                    bf.seek(pre_write_pos)
                    for chunk in iter(lambda: bf.read(8192), b""):
                        self._hasher.update(chunk)
                content_hash = self._hasher.hexdigest()
                size_bytes = self._path.stat().st_size
            except Exception as write_exc:
                # Rollback: truncate file back to pre-write position so no
                # bytes exist on disk without a matching audit record.
                try:
                    file.seek(pre_write_pos)
                    file.truncate(pre_write_pos)
                    file.flush()
                    os.fsync(file.fileno())
                except OSError as rollback_err:
                    raise RuntimeError(
                        f"CSV append failed and rollback also failed — file may be corrupted at byte {pre_write_pos}"
                    ) from rollback_err
                raise write_exc

        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=content_hash,
                size_bytes=size_bytes,
            )
        )

    def _open_file(self, rows: list[dict[str, Any]]) -> None:
        """Open file for append mode.

        Called only from the append-mode path in write(). Write mode uses
        _open_file_write_mode() instead (called after staging succeeds to
        avoid truncating the file before we know the batch is valid).

        In append mode:
        - If file exists with headers: read headers from it, open in append mode
        - If file doesn't exist or is empty: create with headers (like write mode)

        Display Headers:
        When headers mode is CUSTOM or ORIGINAL, the CSV header row uses display
        names but row data uses normalized field names. This is handled by writing
        the header manually and configuring the DictWriter with the original data
        field names.

        Args:
            rows: First batch of rows (used to determine fieldnames if dynamic schema)
        """
        if self._path.exists():
            # Try to read existing headers from file
            with open(self._path, encoding=self._encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=self._delimiter)
                existing_fieldnames = reader.fieldnames

            if existing_fieldnames:
                # Validate headers against explicit schema before opening
                # Dynamic schema = no validation (file headers are authoritative)
                if not self._schema_config.is_observed:
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

                # In append mode with display headers, we need to map existing file headers
                # back to data field names for the DictWriter
                display_map = get_effective_display_headers(self)
                if display_map is not None:
                    # Reverse the display map to get display_name -> data_field
                    reverse_map = {v: k for k, v in display_map.items()}
                    # Map existing headers (display names) back to data field names
                    self._fieldnames = [reverse_map.get(h, h) for h in existing_fieldnames]
                else:
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
                # Initialize hasher with existing file content
                self._hasher = hashlib.sha256()
                with open(self._path, "rb") as bf:
                    for chunk in iter(lambda: bf.read(8192), b""):
                        self._hasher.update(chunk)
                return

        # Append to non-existent/empty file — create with headers
        data_fields, display_fields = self._get_field_names_and_display(rows[0])
        self._fieldnames = data_fields
        self._open_file_write_mode(data_fields, display_fields)

    def _open_file_write_mode(self, data_fields: list[str], display_fields: list[str]) -> None:
        """Open (or create) the file in write mode and write the header row.

        Called AFTER staging succeeds so that the file is never truncated
        before we know the first batch is valid.

        Args:
            data_fields: Field names matching row dict keys (for DictWriter).
            display_fields: Display names for the CSV header row.
        """
        self._fieldnames = data_fields
        self._claim_write_target()
        file_mode = "x" if should_create_exclusively(self._collision_policy) else "w"

        self._file = open(  # noqa: SIM115 - handle kept open for streaming writes, closed in close()
            self._path, file_mode, encoding=self._encoding, newline=""
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            delimiter=self._delimiter,
        )

        # Write header row using display names if configured
        if display_fields != data_fields:
            # Write display headers using csv.writer to handle quoting properly
            # Display names may contain delimiters, quotes, or newlines (e.g., "Amount, USD")
            header_writer = csv.writer(self._file, delimiter=self._delimiter)
            header_writer.writerow(display_fields)
        else:
            # No display mapping - use standard writeheader()
            self._writer.writeheader()

        # Initialize hasher with header bytes
        self._file.flush()
        self._hasher = hashlib.sha256()
        with open(self._path, "rb") as bf:
            for chunk in iter(lambda: bf.read(8192), b""):
                self._hasher.update(chunk)

    def _get_field_names_and_display(self, row: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Get data field names and display names for CSV output.

        Field selection depends on schema mode:
        - fixed: Only declared fields (extras rejected)
        - flexible: Declared fields first, then extras from first row
        - observed: All fields from first row (infer and lock)

        Returns:
            Tuple of (data_fields, display_fields):
            - data_fields: Field names matching row dict keys (for DictWriter)
            - display_fields: Display names for CSV header row
            When no display headers are configured, both lists are identical.
        """
        # Get base field names based on schema mode
        if self._schema_config.is_observed:
            # Observed mode: infer all fields from row keys
            data_fields = list(row.keys())
        elif self._schema_config.fields:
            # Explicit schema: start with declared field names in schema order
            declared_fields = [field_def.name for field_def in self._schema_config.fields]
            declared_set = set(declared_fields)

            if self._schema_config.mode == "flexible":
                # Flexible mode: declared fields first, then extras from row
                extras = [key for key in row if key not in declared_set]
                data_fields = declared_fields + extras
            else:
                # Fixed mode: only declared fields (extras will be rejected by DictWriter)
                data_fields = declared_fields
        else:
            # Fallback (shouldn't happen with valid config): use row keys
            data_fields = list(row.keys())

        # Apply display header mapping if configured
        display_map = get_effective_display_headers(self)
        if display_map is None:
            return data_fields, data_fields

        # Map field names to display names, falling back to original if not mapped
        # This handles transform-added fields that have no original header
        display_fields = [display_map.get(field, field) for field in data_fields]
        return data_fields, display_fields

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        set_resume_field_resolution(self, resolution_mapping)

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
