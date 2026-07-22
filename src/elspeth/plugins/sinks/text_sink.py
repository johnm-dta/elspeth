"""Strict, line-oriented local text sink."""

from __future__ import annotations

import codecs
import keyword
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import CallType, Determinism, PluginSchema
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.sink import OutputValidationResult
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    ResolvedSinkEffectMode,
    RestrictedSinkEffectContext,
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
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import LocalFileSinkConfig, OutputCollisionPolicy
from elspeth.plugins.infrastructure.output_paths import resolve_output_collision_path, validate_output_collision_policy_mode
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


class TextSinkConfig(LocalFileSinkConfig):
    """Configuration for one-field, line-oriented text output."""

    field: str = Field(description="String field written as one line per row.")
    encoding: Literal["utf-8", "ascii", "latin-1", "cp1252"] = Field(
        default="utf-8",
        description="Character encoding used for every emitted line.",
    )
    mode: Literal["write", "append"] = Field(
        default="write",
        description="Write a new output or append lines to an existing output.",
    )

    @field_validator("field")
    @classmethod
    def _validate_field(cls, value: str) -> str:
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError(f"field {value!r} must be a non-keyword Python identifier")
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        try:
            canonical = codecs.lookup(value).name
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {value!r}") from exc
        if canonical not in {"utf-8", "ascii", "iso8859-1", "cp1252"}:
            raise ValueError("encoding must be one of utf-8, ascii, latin-1, or cp1252")
        return value

    @model_validator(mode="after")
    def _validate_collision_mode(self) -> TextSinkConfig:
        validate_output_collision_policy_mode(
            plugin_name="TextSink",
            mode=self.mode,
            collision_policy=self.collision_policy,
        )
        return self


class TextSink(BaseSink):
    """Write one configured string field per canonical LF-delimited record."""

    name = "text"
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:b84f2eb34c0f00c1"
    config_model = TextSinkConfig
    supports_resume = True
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    effect_call_type = CallType.FILESYSTEM
    supported_effect_modes = frozenset({"append", "write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode | None:
        cfg = TextSinkConfig.from_dict(dict(config), plugin_name=cls.name)
        if purpose is SinkEffectExecutionPurpose.RESUME:
            # configure_for_resume() switches the live sink to canonical
            # append before admission is issued; the resolved mode must claim
            # the mode resume actually executes (elspeth-fc9906e398).
            return ResolvedSinkEffectMode("append")
        return ResolvedSinkEffectMode(cfg.mode)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = TextSinkConfig.from_dict(config, plugin_name=self.name)
        self._path = cfg.resolved_path()
        self._requested_path = self._path
        self._field = cfg.field
        self._encoding = cfg.encoding
        self._mode = cfg.mode
        self._collision_policy: OutputCollisionPolicy | None = cfg.collision_policy
        self._schema_config = cfg.schema_config
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "TextSinkRowSchema",
            allow_coercion=False,
        )
        self.input_schema = self._schema_class
        self.declared_required_fields = self._schema_config.get_effective_required_fields() | {self._field}

    def configure_for_resume(self) -> None:
        """Switch a configured write sink to canonical append mode."""
        self._mode = "append"
        self._collision_policy = "append_or_create"

    def validate_output_target(self) -> OutputValidationResult:
        """Validate that an existing target is canonical for safe append."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return OutputValidationResult.success()
        try:
            saw_content = False
            final_character = ""
            with open(self._path, encoding=self._encoding, newline="") as stream:
                for chunk in iter(lambda: stream.read(64 * 1024), ""):
                    if chunk:
                        saw_content = True
                        if "\r" in chunk:
                            return OutputValidationResult.failure(message="Existing text output contains non-canonical CR separators")
                        final_character = chunk[-1]
        except UnicodeError:
            return OutputValidationResult.failure(message=f"Existing text output is not valid {self._encoding}")
        if saw_content and final_character != "\n":
            return OutputValidationResult.failure(message="Existing text output does not end at an LF record boundary")
        return OutputValidationResult.success(target_fields=[self._field])

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
        if type(request.effect_input) is not SinkEffectPipelineMembersInput:
            raise TypeError("TextSink effects require pipeline member input")
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
        accepted: list[int] = []
        diverted: list[int] = []
        diversion_attribution: list[DiversionAttribution] = []

        def chunks() -> Iterator[bytes]:
            if include_baseline and target.exists():
                validation = self.validate_output_target()
                if not validation.valid:
                    raise ValueError(f"Existing text output is incompatible: {validation.error_message}")
                yield from iter_path_chunks(target)
            missing = object()
            for snapshot_member in emitted_members:
                current_member = current_by_effect_id.get(snapshot_member.member_effect_id)
                row = deep_thaw(snapshot_member.row)
                value = row.get(self._field, missing)
                reason: str | None = None
                encoded: bytes | None = None
                if type(value) is not str:
                    reason = f"Text field {self._field!r} must be a string"
                elif "\r" in value or "\n" in value:
                    reason = "Text values cannot contain CR or LF record separators"
                else:
                    try:
                        encoded = (value + "\n").encode(self._encoding)
                    except UnicodeEncodeError:
                        reason = f"Text value is not representable in configured codec {self._encoding}"
                if reason is not None:
                    if current_member is None:
                        raise ValueError(f"Predecessor text snapshot is incompatible: {reason}")
                    self._divert_row(row, row_index=current_member.ordinal, reason=reason)
                    diverted.append(current_member.ordinal)
                    diversion_attribution.append(build_diversion_attribution(ordinal=current_member.ordinal, reason=reason))
                    continue
                assert encoded is not None
                if current_member is not None:
                    accepted.append(current_member.ordinal)
                yield encoded

        return prepare_local_effect(
            effect_id=request.effect_id,
            input_kind=request.input_kind,
            inspection=request.inspection,
            chunks=chunks(),
            row_count=len(members),
            accepted_ordinals=lambda: accepted,
            diverted_ordinals=lambda: diverted,
            encoding=self._encoding,
            format_name="text",
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
        raise RuntimeError("TextSink publication requires the recoverable sink effect coordinator") from None

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
                plugin_name=cls.name,
                issue_code=None,
                summary="Write one configured string field per row as canonical LF-delimited text.",
                composer_hints=(
                    "Set field to the configured field whose value should become each output line; every accepted value must be a string.",
                    "Text values containing CR or LF are rejected rather than escaped, so one input row always maps to exactly one output record.",
                    "Choose only utf-8, ascii, latin-1, or cp1252; values not representable in the configured encoding are diverted without leaking their content.",
                    "Use mode: append with collision_policy: append_or_create for append/resume; existing bytes must decode cleanly and end with canonical LF.",
                    "Text is not a generic failure sink because it deliberately writes only one configured field and cannot preserve a rejected row losslessly.",
                ),
            )
        return None
