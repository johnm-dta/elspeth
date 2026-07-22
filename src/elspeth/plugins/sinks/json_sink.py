"""JSON sink plugin for ELSPETH.

Writes rows to JSON files. Supports JSON array and JSONL formats.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import codecs
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

from elspeth.contracts import CallType, Determinism, PluginSchema
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import SinkEffectCapabilityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.header_modes import HeaderMode
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
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
    apply_display_headers,
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


def _iter_file_without_suffix(path: Path, suffix: bytes) -> Iterator[bytes]:
    """Stream a file except for one exact structural suffix."""
    pending = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            combined = pending + chunk
            if len(combined) > len(suffix):
                yield combined[: -len(suffix)]
                pending = combined[-len(suffix) :]
            else:
                pending = combined
    if pending != suffix:
        raise ValueError("Existing JSON array does not have the expected terminal boundary")


def _indent_json_value(serialized: str, indent: int) -> str:
    prefix = " " * indent
    return "\n".join(prefix + line for line in serialized.splitlines())


class JSONSinkConfig(SinkPathConfig):
    """Configuration for JSON sink plugin.

    Inherits from SinkPathConfig, which provides:
    - Path handling (from PathConfig)
    - Schema configuration (from DataPluginConfig)
    - Header output mode (headers: normalized | original | {mapping})
    """

    format: Literal["json", "jsonl"] | None = Field(
        default=None,
        description="Output JSON format. When omitted, the sink auto-detects JSONL from a .jsonl filename and JSON otherwise.",
    )
    indent: int | None = Field(default=None, description="Indentation level for JSON array output; null writes compact JSON.")
    encoding: str = Field(default="utf-8", description="Text encoding used when writing JSON output.")
    mode: Literal["write", "append"] = Field(
        default="write",
        description="Whether to create/replace the JSON output file or append JSONL rows.",
    )

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
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:8eaa85553aae688d"
    config_model = JSONSinkConfig
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    effect_call_type = CallType.FILESYSTEM
    supported_effect_modes = frozenset({"append", "write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS, SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT})
    supported_audit_export_formats = frozenset({AuditExportFormat.JSON})

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode | None:
        cfg = JSONSinkConfig.from_dict(dict(config), plugin_name=cls.name)
        if purpose is SinkEffectExecutionPurpose.AUDIT_EXPORT:
            cls._validate_audit_export_config(cfg)
        if purpose is SinkEffectExecutionPurpose.RESUME:
            # configure_for_resume() switches the live sink to canonical
            # append before admission is issued; the resolved mode must claim
            # the mode resume actually executes (elspeth-fc9906e398). A JSON
            # array target never reaches admission under resume: its
            # configure_for_resume() refusal aborts the flow first.
            return ResolvedSinkEffectMode("append")
        return ResolvedSinkEffectMode(cfg.mode)

    @classmethod
    def _validate_audit_export_config(cls, cfg: JSONSinkConfig) -> None:
        if cfg.mode != "write" or cfg.collision_policy not in {None, "fail_if_exists"}:
            raise SinkEffectCapabilityError("JSON audit export requires mode='write' and collision policy 'fail_if_exists' or null")

    def _validate_sink_effect_capability_configuration(
        self,
        *,
        mode: str,
        required_input_kind: SinkEffectInputKind,
    ) -> None:
        if mode != self._mode:
            raise SinkEffectCapabilityError("JSON sink effect mode does not match configured mode")
        if required_input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT:
            self._validate_audit_export_config(JSONSinkConfig.from_dict(dict(self.config), plugin_name=self.name))

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
            expected = [display_name_for(display_map, f) for f in expected_normalized]
        elif self._headers_mode == HeaderMode.ORIGINAL:
            return OutputValidationResult.failure(
                message="JSONL headers: original requires source field resolution before resume validation",
                target_fields=existing,
                schema_fields=expected_normalized,
            )
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
        if self._mode != "append" and not plugin_preflight_mode_enabled():
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

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del ctx
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
        effect_input = request.effect_input
        if type(effect_input) is SinkEffectAuditExportSnapshotInput:
            if effect_input.export_format is not AuditExportFormat.JSON:
                raise TypeError("JSONSink audit effects require a JSON audit snapshot")

            def audit_export_chunks() -> Iterator[bytes]:
                yield from effect_input.reader.iter_verified_chunks()
                manifest = effect_input.reader.read_verified_signed_manifest()
                if manifest.endswith(b"\n"):
                    raise ValueError("audit export final manifest must not carry a trailing newline")
                yield manifest

            return prepare_local_effect(
                effect_id=request.effect_id,
                input_kind=request.input_kind,
                inspection=request.inspection,
                chunks=audit_export_chunks(),
                row_count=0,
                accepted_ordinals=(),
                diverted_ordinals=(),
                encoding="utf-8",
                format_name="audit_export_jsonl",
                stream_sequence=0,
            )
        if type(effect_input) is not SinkEffectPipelineMembersInput:
            raise TypeError("JSONSink effects require a closed supported effect input")

        members = effect_input.members
        target_snapshot_members = effect_input.target_snapshot_members
        current_by_effect_id = {member.member_effect_id: member for member in members}
        target = Path(str(request.inspection.evidence["target_path"]))
        predecessor_declared = bool(request.inspection.evidence["predecessor_declared"])
        include_baseline, emitted_members = continuation_emission(
            append_mode=self._mode == "append",
            predecessor_declared=predecessor_declared,
            current_member_effect_ids=current_by_effect_id.keys(),
            target_snapshot_members=target_snapshot_members,
        )
        accepted: list[int] = []
        diverted: list[int] = []
        diversion_attribution: list[DiversionAttribution] = []

        def serialized_rows() -> Iterator[tuple[int, str]]:
            source_rows = [deep_thaw(member.row) for member in emitted_members]
            output_rows = apply_display_headers(self, source_rows)
            for snapshot_member, original, output in zip(emitted_members, source_rows, output_rows, strict=True):
                current_member = current_by_effect_id.get(snapshot_member.member_effect_id)
                try:
                    serialized = json.dumps(output, indent=self._indent if self._format == "json" else None, allow_nan=False)
                except (ValueError, TypeError) as exc:
                    reason = f"JSON serialization failed: {exc}"
                    if current_member is None:
                        raise ValueError(f"Predecessor JSON snapshot is incompatible: {reason}") from exc
                    self._divert_row(original, row_index=current_member.ordinal, reason=reason)
                    diverted.append(current_member.ordinal)
                    diversion_attribution.append(build_diversion_attribution(ordinal=current_member.ordinal, reason=reason))
                    continue
                if current_member is not None:
                    accepted.append(current_member.ordinal)
                serialized.encode(self._encoding)
                yield snapshot_member.ordinal, serialized

        def jsonl_chunks() -> Iterator[bytes]:
            encoder = codecs.getincrementalencoder(self._encoding)()
            if include_baseline and target.exists():
                validation = self.validate_output_target()
                if not validation.valid:
                    raise ValueError(f"Existing JSONL output is incompatible: {validation.error_message}")
                yield from iter_path_chunks(target)
                encoder.setstate(0)
            for _ordinal, serialized in serialized_rows():
                yield encoder.encode(serialized + "\n")
            final = encoder.encode("", final=True)
            if final:
                yield final

        def json_array_chunks() -> Iterator[bytes]:
            encoder = codecs.getincrementalencoder(self._encoding)()
            iterator = iter(serialized_rows())
            try:
                _first_ordinal, first = next(iterator)
            except StopIteration:
                if include_baseline and target.exists():
                    yield from iter_path_chunks(target)
                else:
                    yield encoder.encode("[]", final=True)
                return

            baseline_nonempty = False
            if include_baseline and target.exists():
                size = target.stat().st_size
                if size > 64:
                    baseline_nonempty = True
                else:
                    with target.open(encoding=self._encoding) as stream:
                        baseline_nonempty = stream.read().strip() != "[]"
            if baseline_nonempty:
                suffix_encoder = codecs.getincrementalencoder(self._encoding)()
                suffix_encoder.setstate(0)
                suffix = suffix_encoder.encode("]" if self._indent is None else "\n]", final=True)
                yield from _iter_file_without_suffix(target, suffix)
                encoder.setstate(0)
                yield encoder.encode(", " if self._indent is None else ",\n")
            else:
                yield encoder.encode("[" if self._indent is None else "[\n")

            if self._indent is None:
                yield encoder.encode(first)
            else:
                yield encoder.encode(_indent_json_value(first, self._indent))
            for _ordinal, serialized in iterator:
                yield encoder.encode(", " if self._indent is None else ",\n")
                if self._indent is None:
                    yield encoder.encode(serialized)
                else:
                    yield encoder.encode(_indent_json_value(serialized, self._indent))
            yield encoder.encode("]" if self._indent is None else "\n]", final=True)

        return prepare_local_effect(
            effect_id=request.effect_id,
            input_kind=request.input_kind,
            inspection=request.inspection,
            chunks=jsonl_chunks() if self._format == "jsonl" else json_array_chunks(),
            row_count=len(members),
            accepted_ordinals=lambda: accepted,
            diverted_ordinals=lambda: diverted,
            encoding=self._encoding,
            format_name=self._format,
            stream_sequence=1 if predecessor_declared else 0,
            diversion_attribution=lambda: diversion_attribution,
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del ctx
        return commit_local_effect(plan)

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del ctx
        return reconcile_local_effect(plan)

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        del rows, ctx
        raise RuntimeError("JSONSink publication requires the recoverable sink effect coordinator") from None

    def flush(self) -> None:
        """No-op: ``commit_effect`` performs synchronous publication.

        Direct ``write`` publication is forbidden. The recoverable sink-effect
        coordinator owns inspection, intent recording, synchronous commit, and
        reconciliation, so there is no independent buffered state to flush.
        """
        pass

    def close(self) -> None:
        """No-op: file handles are opened and closed inside the effect commit path."""

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        set_resume_field_resolution(self, resolution_mapping)

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name="json",
                issue_code=None,
                summary="Write rows as JSON-array or JSONL (newline-delimited). Configurable collision_policy, encoding, and on_write_failure routing.",
                composer_hints=(
                    "Choose format: 'jsonl' for resumable output. JSON-array rewrites the entire file on every checkpoint — not resumable.",
                    "collision_policy: 'fail_if_exists', 'auto_increment', or 'append_or_create' (mode: append only). REQUIRED explicitly with mode: write — the composer rejects an implicit collision policy; 'auto_increment' is the safe default.",
                    "JSON sink writes the row it receives; schema, format, sink name, and output name do not drop fields. Use field_mapper before the sink when the user wants to remove or whitelist fields.",
                    "For web_scrape results saved without raw page bodies, route the final path through field_mapper(select_only=true) before this sink; a sink named cleanup is not a cleanup transform.",
                    "on_write_failure is REQUIRED (no default): set 'discard' (drop with an audit record) or a quarantine sink name so single-row write errors don't crash the run; omitting it fails validation.",
                    "path templating supports {run_id}, {date}, {sink_name} — use these to avoid collision in concurrent or scheduled runs.",
                ),
            )
        return None

    @classmethod
    def get_post_call_hints(
        cls,
        *,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        hints: list[str] = []
        if "format" in config_snapshot and config_snapshot["format"] == "json":
            hints.append(
                "format: 'json' (array mode) rewrites the entire file at every checkpoint and is not resumable. "
                "If the run might be interrupted or long-running, switch to format: 'jsonl'."
            )
        if "on_write_failure" not in config_snapshot:
            hints.append(
                "on_write_failure is not set, but it is REQUIRED — there is no default. Set it to 'discard' to drop failed rows (with an audit record) or to a quarantine sink name to divert and audit them; a run without it fails validation."
            )
        return tuple(hints)
