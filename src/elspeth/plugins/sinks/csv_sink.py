"""CSV sink plugin for ELSPETH.

Writes rows to CSV files with content hashing for audit integrity.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

from __future__ import annotations

import codecs
import csv
import io
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import CallType, Determinism, PluginSchema
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import SinkEffectCapabilityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    AuditExportFormat,
    ResolvedSinkEffectMode,
    RestrictedSinkEffectContext,
    SinkEffectAuditExportSnapshotInput,
    SinkEffectCommitResult,
    SinkEffectExecutionPurpose,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionRequest,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from elspeth.contracts.contexts import SinkContext
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import OutputCollisionPolicy, SinkPathConfig
from elspeth.plugins.infrastructure.display_headers import (
    display_name_for,
    get_effective_display_headers,
    init_display_headers,
    set_resume_field_resolution,
)
from elspeth.plugins.infrastructure.output_paths import (
    resolve_output_collision_path,
    validate_output_collision_policy_mode,
)
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode_enabled
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.sinks._audit_export_bundle_effects import (
    commit_audit_export_bundle,
    inspect_audit_export_bundle,
    prepare_audit_export_bundle,
    reconcile_audit_export_bundle,
)
from elspeth.plugins.sinks._diversion_attribution import DiversionAttribution, build_diversion_attribution
from elspeth.plugins.sinks._local_file_effects import (
    commit_local_effect,
    continuation_emission,
    inspect_local_effect,
    iter_path_chunks,
    predecessor_local_path,
    prepare_local_effect,
    reconcile_local_effect,
)


class CSVSinkConfig(SinkPathConfig):
    """Configuration for CSV sink plugin.

    Inherits from SinkPathConfig, which provides:
    - Path handling (from PathConfig)
    - Schema configuration (from DataPluginConfig)
    - Header output mode (headers: normalized | original | {mapping})
    """

    delimiter: str = Field(default=",", description="Single-character delimiter used when writing CSV fields.")
    encoding: str = Field(default="utf-8", description="Text encoding used when writing the CSV file.")
    mode: Literal["write", "append"] = Field(default="write", description="Whether to create/replace the CSV file or append rows.")

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
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:a67f1d6397cb04ee"
    config_model = CSVSinkConfig
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    effect_call_type = CallType.FILESYSTEM
    supported_effect_modes = frozenset({"append", "write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS, SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT})
    supported_audit_export_formats = frozenset({AuditExportFormat.CSV})

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode | None:
        cfg = CSVSinkConfig.from_dict(dict(config), plugin_name=cls.name)
        if purpose is SinkEffectExecutionPurpose.AUDIT_EXPORT:
            cls._validate_audit_export_config(cfg)
        if purpose is SinkEffectExecutionPurpose.RESUME:
            # configure_for_resume() switches the live sink to canonical
            # append before admission is issued; the resolved mode must claim
            # the mode resume actually executes (elspeth-fc9906e398).
            return ResolvedSinkEffectMode("append")
        return ResolvedSinkEffectMode(cfg.mode)

    @classmethod
    def _validate_audit_export_config(cls, cfg: CSVSinkConfig) -> None:
        if cfg.mode != "write" or cfg.collision_policy not in {None, "fail_if_exists"}:
            raise SinkEffectCapabilityError(
                "CSV audit export requires mode='write' and create-only collision policy 'fail_if_exists' or null"
            )

    def _validate_sink_effect_capability_configuration(
        self,
        *,
        mode: str,
        required_input_kind: SinkEffectInputKind,
    ) -> None:
        if mode != self._mode:
            raise SinkEffectCapabilityError("CSV sink effect mode does not match configured mode")
        if required_input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT:
            self._validate_audit_export_config(CSVSinkConfig.from_dict(dict(self.config), plugin_name=self.name))

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
            expected = [display_name_for(display_map, f) for f in expected_normalized]
        elif self._headers_mode == HeaderMode.ORIGINAL:
            return OutputValidationResult.failure(
                message="CSV headers: original requires source field resolution before resume validation",
                target_fields=existing,
                schema_fields=expected_normalized,
            )
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
        if self._mode != "append" and not plugin_preflight_mode_enabled():
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

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del ctx
        if request.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT:
            self._path = self._requested_path
            return inspect_audit_export_bundle(target_path=self._path, request=request)
        predecessor_path = predecessor_local_path(request)
        if predecessor_path is not None:
            self._path = predecessor_path
        elif self._mode != "append":
            self._path = resolve_output_collision_path(self._requested_path, self._collision_policy)
        return inspect_local_effect(target_path=self._path, request=request)

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        if type(request.effect_input) is SinkEffectAuditExportSnapshotInput:
            return prepare_audit_export_bundle(target_path=self._requested_path, request=request)
        if type(request.effect_input) is not SinkEffectPipelineMembersInput:
            raise TypeError("CSVSink effects require pipeline member input")
        members = request.effect_input.members
        target_snapshot_members = request.effect_input.target_snapshot_members
        current_by_effect_id = {member.member_effect_id: member for member in members}
        target = Path(str(request.inspection.evidence["target_path"]))
        predecessor_declared = bool(request.inspection.evidence["predecessor_declared"])
        include_baseline, emitted_members = continuation_emission(
            append_mode=self._mode == "append",
            predecessor_declared=predecessor_declared,
            current_member_effect_ids=current_by_effect_id.keys(),
            target_snapshot_members=target_snapshot_members,
        )
        baseline_nonempty = include_baseline and target.exists() and target.stat().st_size > 0
        rows = [deep_thaw(member.row) for member in emitted_members]

        if baseline_nonempty:
            validation = self.validate_output_target()
            if not validation.valid:
                raise ValueError(f"Existing CSV output is incompatible: {validation.error_message}")
            with target.open(encoding=self._encoding, newline="") as stream:
                existing_headers = list(csv.DictReader(stream, delimiter=self._delimiter).fieldnames or ())
            display_map = get_effective_display_headers(self)
            if display_map is not None:
                reverse_map = {value: key for key, value in display_map.items()}
                data_fields = [display_name_for(reverse_map, header) for header in existing_headers]
            else:
                data_fields = existing_headers
            display_fields = existing_headers
        else:
            data_fields, display_fields = self._get_field_names_and_display(rows[0])

        accepted: list[int] = []
        diverted: list[int] = []
        diversion_attribution: list[DiversionAttribution] = []

        def header_text() -> str:
            buffer = io.StringIO(newline="")
            csv.writer(buffer, delimiter=self._delimiter).writerow(display_fields)
            return buffer.getvalue()

        def chunks() -> Iterator[bytes]:
            encoder = codecs.getincrementalencoder(self._encoding)()
            if baseline_nonempty:
                yield from iter_path_chunks(target)
                encoder.setstate(0)
            else:
                yield encoder.encode(header_text())
            locked_fields = set(data_fields)
            for snapshot_member, row in zip(emitted_members, rows, strict=True):
                current_member = current_by_effect_id.get(snapshot_member.member_effect_id)
                extra_fields = set(row) - locked_fields
                if extra_fields:
                    reason = "CSV row contains fields outside the established columns: " + ", ".join(
                        sorted(str(field) for field in extra_fields)
                    )
                    if current_member is None:
                        raise ValueError(f"Predecessor CSV snapshot is incompatible: {reason}")
                    self._divert_row(row, row_index=current_member.ordinal, reason=reason)
                    diverted.append(current_member.ordinal)
                    diversion_attribution.append(build_diversion_attribution(ordinal=current_member.ordinal, reason=reason))
                    continue
                row_buffer = io.StringIO(newline="")
                writer = csv.DictWriter(row_buffer, fieldnames=data_fields, delimiter=self._delimiter)
                try:
                    writer.writerow(row)
                    row_text = row_buffer.getvalue()
                    row_text.encode(self._encoding)
                except UnicodeEncodeError as exc:
                    reason = f"CSV encoding ({self._encoding}) failed: {exc}"
                    if current_member is None:
                        raise ValueError(f"Predecessor CSV snapshot is incompatible: {reason}") from exc
                    self._divert_row(
                        row,
                        row_index=current_member.ordinal,
                        reason=reason,
                    )
                    diverted.append(current_member.ordinal)
                    diversion_attribution.append(build_diversion_attribution(ordinal=current_member.ordinal, reason=reason))
                    continue
                except csv.Error as exc:
                    reason = f"CSV serialization failed: {exc}"
                    if current_member is None:
                        raise ValueError(f"Predecessor CSV snapshot is incompatible: {reason}") from exc
                    self._divert_row(row, row_index=current_member.ordinal, reason=reason)
                    diverted.append(current_member.ordinal)
                    diversion_attribution.append(build_diversion_attribution(ordinal=current_member.ordinal, reason=reason))
                    continue
                if current_member is not None:
                    accepted.append(current_member.ordinal)
                yield encoder.encode(row_text)
            final = encoder.encode("", final=True)
            if final:
                yield final

        return prepare_local_effect(
            effect_id=request.effect_id,
            input_kind=request.input_kind,
            inspection=request.inspection,
            chunks=chunks(),
            row_count=len(members),
            accepted_ordinals=lambda: accepted,
            diverted_ordinals=lambda: diverted,
            encoding=self._encoding,
            format_name="csv",
            stream_sequence=1 if predecessor_declared else 0,
            diversion_attribution=lambda: diversion_attribution,
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del ctx
        if plan.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT:
            return commit_audit_export_bundle(plan)
        return commit_local_effect(plan)

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del ctx
        if plan.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT:
            return reconcile_audit_export_bundle(plan)
        return reconcile_local_effect(plan)

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        del rows, ctx
        raise RuntimeError("CSVSink publication requires the recoverable sink effect coordinator") from None

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

        from elspeth.contracts.header_modes import HeaderMode

        if self._headers_mode == HeaderMode.CUSTOM:
            for field in data_fields:
                if field not in display_map:
                    raise ValueError(
                        f"CUSTOM header mode has no mapping for field '{field}'. "
                        f"All fields must be explicitly mapped — silent fallback to normalized "
                        f"names risks data corruption in external system handover. "
                        f"Mapped fields: {sorted(display_map.keys())}"
                    )

        # ORIGINAL mode falls back to the normalized field name for
        # transform-added fields that have no source header.
        display_fields = [display_name_for(display_map, field) for field in data_fields]
        return data_fields, display_fields

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        set_resume_field_resolution(self, resolution_mapping)

    def flush(self) -> None:
        """No-op: ``commit_effect`` performs synchronous publication.

        Direct ``write`` publication is forbidden. The recoverable sink-effect
        coordinator owns inspection, intent recording, synchronous commit, and
        reconciliation, so there is no independent buffered state to flush.
        """
        pass

    def close(self) -> None:
        """No-op: file handles are opened and closed inside the effect commit path."""

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name="csv",
                issue_code=None,
                summary="Write rows as CSV. Configurable delimiter, encoding, mode, collision_policy, and header-display overrides.",
                composer_hints=(
                    "delimiter (default ',', single character) is an operator concern — pick for the consuming tool (Excel: ','; analytics tools may prefer '\\t'). Quoting is fixed (standard csv quoting); only delimiter and encoding are configurable.",
                    "collision_policy: 'fail_if_exists', 'auto_increment', or 'append_or_create' (only with mode: append). Unset is the default and OVERWRITES/truncates an existing file — set it deliberately to protect prior runs.",
                    "Header row is written once at start. Resume appends must use the same column order; pin headers explicitly with headers when schema can evolve.",
                    "on_write_failure is REQUIRED (no default): set 'discard' (drop with an audit record) or a quarantine sink name so per-row write errors don't crash the run; omitting it fails validation.",
                ),
            )
        return None
