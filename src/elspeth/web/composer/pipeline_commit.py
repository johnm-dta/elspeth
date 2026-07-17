"""Lock-assuming canonical pipeline proposal commit preparation contracts."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Engine

from elspeth.contracts.composer_audit import (
    ComposerToolInvocation,
    ComposerToolStatus,
    PipelineDispatchAuditPayload,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder, begin_dispatch, dispatch_with_audit
from elspeth.web.composer.pipeline_proposal import AbsentBase, PlannerSurface, PresentBase, composition_content_hash
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools._common import RuntimePreflight, ToolContext, ToolResult
from elspeth.web.composer.tools._dispatch import execute_tool
from elspeth.web.composer.tools.sessions import build_set_pipeline_candidate
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.protocol import AuthoritativePipelineProposal

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PipelineDispatchAuditBinding:
    """Smallest durable binding exposed by the current redacted audit store."""

    tool_call_id: str
    tool_name: str
    status: ComposerToolStatus
    arguments_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        if type(self.tool_call_id) is not str or not self.tool_call_id.strip():
            raise AuditIntegrityError("pipeline dispatch tool_call_id must be a non-empty exact string")
        if self.tool_name != "set_pipeline":
            raise AuditIntegrityError("pipeline dispatch tool_name must be set_pipeline")
        if type(self.status) is not ComposerToolStatus or self.status is not ComposerToolStatus.SUCCESS:
            raise AuditIntegrityError("pipeline dispatch binding requires a successful dispatch")
        for name in ("arguments_hash", "result_hash"):
            value = getattr(self, name)
            if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
                raise AuditIntegrityError(f"pipeline dispatch {name} must be a SHA-256 hash")

    @classmethod
    def from_invocation(cls, invocation: ComposerToolInvocation) -> PipelineDispatchAuditBinding:
        if type(invocation) is not ComposerToolInvocation:
            raise TypeError("invocation must be an exact ComposerToolInvocation")
        if invocation.result_hash is None:
            raise AuditIntegrityError("successful pipeline dispatch is missing result_hash")
        return cls(
            tool_call_id=invocation.tool_call_id,
            tool_name=invocation.tool_name,
            status=invocation.status,
            arguments_hash=invocation.arguments_hash,
            result_hash=invocation.result_hash,
        )

    @classmethod
    def from_persisted_envelope(cls, envelope: Mapping[str, Any]) -> PipelineDispatchAuditBinding:
        """Restore the binding from the exact redacted envelope written to chat audit."""
        if type(envelope) is not dict or envelope.get("_kind") != "audit":
            raise AuditIntegrityError("persisted pipeline dispatch envelope is malformed")
        invocation = envelope.get("invocation")
        if type(invocation) is not dict:
            raise AuditIntegrityError("persisted pipeline dispatch invocation is malformed")
        arguments_canonical = invocation.get("arguments_canonical")
        result_canonical = invocation.get("result_canonical")
        raw_status = invocation.get("status")
        tool_call_id = invocation.get("tool_call_id")
        tool_name = invocation.get("tool_name")
        if type(arguments_canonical) is not str or type(result_canonical) is not str:
            raise AuditIntegrityError("persisted successful pipeline dispatch canonical payloads are malformed")
        if type(raw_status) is not str or type(tool_call_id) is not str or type(tool_name) is not str:
            raise AuditIntegrityError("persisted pipeline dispatch scalar fields are malformed")
        try:
            arguments_hash = stable_hash(json.loads(arguments_canonical))
            result_hash = stable_hash(json.loads(result_canonical))
            status = ComposerToolStatus(raw_status)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AuditIntegrityError("persisted pipeline dispatch payload is malformed") from exc
        if invocation.get("arguments_hash") != arguments_hash or invocation.get("result_hash") != result_hash:
            raise AuditIntegrityError("persisted pipeline dispatch canonical hashes are malformed")
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            arguments_hash=arguments_hash,
            result_hash=result_hash,
        )

    def to_dict(self) -> PipelineDispatchAuditPayload:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "arguments_hash": self.arguments_hash,
            "result_hash": self.result_hash,
        }


class PipelineCommitError(RuntimeError):
    """Leak-safe pipeline preparation failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        invocation: ComposerToolInvocation | None = None,
        dispatch: PipelineDispatchAuditBinding | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.invocation = invocation
        self.dispatch = dispatch


class PipelineCommitMismatchError(PipelineCommitError):
    """Candidate and audited executor produced different composition content."""

    def __init__(self, invocation: ComposerToolInvocation | None, dispatch: PipelineDispatchAuditBinding) -> None:
        super().__init__(
            "pipeline candidate/executor content mismatch",
            code="CANDIDATE_EXECUTOR_MISMATCH",
            invocation=invocation,
            dispatch=dispatch,
        )


@dataclass(frozen=True, slots=True)
class PipelineCommitConfig:
    data_dir: str
    session_engine: Engine | None
    secret_service: WebSecretResolver | None
    user_id: str | None
    user_message_content: str | None
    max_blob_storage_per_session_bytes: int
    runtime_preflight: RuntimePreflight | None
    timeout_seconds: float

    def __post_init__(self) -> None:
        if type(self.data_dir) is not str or not self.data_dir.strip():
            raise ValueError("data_dir must be a non-empty exact string")
        if self.user_id is not None and (type(self.user_id) is not str or not self.user_id.strip()):
            raise ValueError("user_id must be a non-empty exact string or None")
        if self.user_message_content is not None and type(self.user_message_content) is not str:
            raise TypeError("user_message_content must be an exact string or None")
        if type(self.max_blob_storage_per_session_bytes) is not int or self.max_blob_storage_per_session_bytes <= 0:
            raise ValueError("max_blob_storage_per_session_bytes must be a positive exact integer")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int | float):
            raise TypeError("timeout_seconds must be a finite positive number")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")


@dataclass(frozen=True, slots=True)
class PreparedPipelineCommit:
    result: ToolResult
    candidate_content_hash: str
    executor_content_hash: str
    invocation: ComposerToolInvocation
    dispatch: PipelineDispatchAuditBinding

    def __post_init__(self) -> None:
        if type(self.result) is not ToolResult:
            raise TypeError("result must be an exact ToolResult")
        if type(self.invocation) is not ComposerToolInvocation:
            raise TypeError("invocation must be an exact ComposerToolInvocation")
        if type(self.dispatch) is not PipelineDispatchAuditBinding:
            raise TypeError("dispatch must be an exact PipelineDispatchAuditBinding")
        for name in ("candidate_content_hash", "executor_content_hash"):
            value = getattr(self, name)
            if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
                raise AuditIntegrityError(f"{name} must be a SHA-256 hash")


@dataclass(frozen=True, slots=True)
class RecoveredPipelineCommit:
    """Candidate rebuilt against one durable executor-content binding."""

    result: ToolResult
    candidate_content_hash: str
    executor_content_hash: str
    dispatch: PipelineDispatchAuditBinding


def _bind_executor_content_hash(
    invocation: ComposerToolInvocation,
    *,
    executor_content_hash: str,
) -> ComposerToolInvocation:
    if invocation.result_canonical is None:
        raise AuditIntegrityError("successful pipeline dispatch is missing canonical result")
    try:
        payload = json.loads(invocation.result_canonical)
    except json.JSONDecodeError as exc:
        raise AuditIntegrityError("successful pipeline dispatch result is malformed") from exc
    if type(payload) is not dict:
        raise AuditIntegrityError("successful pipeline dispatch result must be a mapping")
    if "pipeline_content_hash" in payload or "pipeline_content_hash_schema" in payload:
        raise AuditIntegrityError("pipeline dispatch result uses a reserved content binding field")
    payload["pipeline_content_hash_schema"] = "composer.pipeline-dispatch-result.v1"
    payload["pipeline_content_hash"] = executor_content_hash
    result_canonical = canonical_json(payload)
    return replace(
        invocation,
        result_canonical=result_canonical,
        result_hash=stable_hash(payload),
    )


async def prepare_pipeline_proposal_commit(
    *,
    authority: AuthoritativePipelineProposal,
    current_state: CompositionState,
    current_state_id: UUID | None,
    policy_catalog: PolicyCatalogView,
    plugin_snapshot: PluginAvailabilitySnapshot,
    config: PipelineCommitConfig,
    recorder: BufferingRecorder,
    actor: str,
    recovery_dispatch: PipelineDispatchAuditBinding | None = None,
    recovery_executor_content_hash: str | None = None,
) -> PreparedPipelineCommit | RecoveredPipelineCommit:
    """Revalidate and audited-dispatch exact arguments; never settle state."""
    if type(authority) is not AuthoritativePipelineProposal:
        raise TypeError("authority must be an exact AuthoritativePipelineProposal")
    if authority.proposal.surface in {PlannerSurface.GUIDED_STAGED, PlannerSurface.TUTORIAL_PROFILE}:
        raise PipelineCommitError("generic route cannot settle staged pipeline proposals", code="SURFACE_REQUIRES_GUIDED")
    if authority.row.status != "pending":
        raise PipelineCommitError("pipeline proposal is not pending", code="NOT_PENDING")
    if policy_catalog.snapshot is not plugin_snapshot:
        raise ValueError("plugin_snapshot_catalog_mismatch")
    if type(authority.proposal.base) is AbsentBase:
        if current_state_id is not None:
            raise PipelineCommitError("pipeline proposal base changed", code="BASE_CONFLICT")
    elif type(authority.proposal.base) is PresentBase:
        if current_state_id != authority.proposal.base.state_id:
            raise PipelineCommitError("pipeline proposal base changed", code="BASE_CONFLICT")
        if composition_content_hash(current_state) != authority.proposal.base.composition_content_hash:
            raise PipelineCommitError("pipeline proposal base content changed", code="BASE_CONFLICT")
    else:
        raise AuditIntegrityError("pipeline proposal base is malformed")

    pipeline_arguments = deep_thaw(authority.proposal.pipeline)
    if type(pipeline_arguments) is not dict:
        raise AuditIntegrityError("authoritative pipeline arguments must thaw to an exact mapping")
    if stable_hash(pipeline_arguments) != authority.row.tool_arguments_hash:
        raise AuditIntegrityError("authoritative pipeline arguments do not match the proposal row")

    prior_validation = policy_catalog.validate_composition_state(current_state).validation
    context = ToolContext(
        catalog=policy_catalog,
        plugin_snapshot=plugin_snapshot,
        data_dir=config.data_dir,
        require_data_dir_for_paths=True,
        session_engine=config.session_engine,
        session_id=str(authority.row.session_id),
        secret_service=config.secret_service,
        user_id=config.user_id,
        baseline=current_state,
        current_validation=prior_validation,
        runtime_preflight=config.runtime_preflight,
        max_blob_storage_per_session_bytes=config.max_blob_storage_per_session_bytes,
        user_message_id=str(authority.row.user_message_id) if authority.row.user_message_id is not None else None,
        user_message_content=config.user_message_content,
        composer_model_identifier=authority.row.composer_model_identifier,
        composer_model_version=authority.row.composer_model_version,
        composer_provider=authority.row.composer_provider,
        composer_skill_hash=authority.row.composer_skill_hash,
        tool_arguments_hash=authority.row.tool_arguments_hash,
    )

    async def bounded(func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(run_sync_in_worker(func, *args, **kwargs), timeout=config.timeout_seconds)
        except TimeoutError as exc:
            raise PipelineCommitError("pipeline commit preparation timed out", code="TIMEOUT") from exc

    candidate = await bounded(
        build_set_pipeline_candidate,
        pipeline_arguments,
        current_state,
        context,
    )
    if not candidate.acceptable or candidate.prepared_inline_blob is not None:
        raise PipelineCommitError("pipeline proposal failed current candidate validation", code="VALIDATION_FAILED")
    candidate_hash = composition_content_hash(candidate.result.updated_state)

    if (recovery_dispatch is None) != (recovery_executor_content_hash is None):
        raise AuditIntegrityError("pipeline recovery dispatch and executor hash must be supplied together")
    if recovery_dispatch is not None:
        assert recovery_executor_content_hash is not None
        if candidate_hash != recovery_executor_content_hash:
            raise PipelineCommitMismatchError(None, recovery_dispatch)
        return RecoveredPipelineCommit(
            result=candidate.result,
            candidate_content_hash=candidate_hash,
            executor_content_hash=recovery_executor_content_hash,
            dispatch=recovery_dispatch,
        )

    audit = begin_dispatch(
        authority.row.tool_call_id,
        "set_pipeline",
        pipeline_arguments,
        version_before=current_state.version,
        actor=actor,
    )
    before_count = len(recorder.invocations)

    async def execute_exact() -> ToolResult:
        return cast(
            ToolResult,
            await bounded(
                execute_tool,
                "set_pipeline",
                pipeline_arguments,
                current_state,
                policy_catalog,
                plugin_snapshot=plugin_snapshot,
                data_dir=config.data_dir,
                session_engine=config.session_engine,
                session_id=str(authority.row.session_id),
                secret_service=config.secret_service,
                user_id=config.user_id,
                baseline=current_state,
                prior_validation=prior_validation,
                runtime_preflight=config.runtime_preflight,
                max_blob_storage_per_session_bytes=config.max_blob_storage_per_session_bytes,
                user_message_id=str(authority.row.user_message_id) if authority.row.user_message_id is not None else None,
                user_message_content=config.user_message_content,
                composer_model_identifier=authority.row.composer_model_identifier,
                composer_model_version=authority.row.composer_model_version,
                composer_provider=authority.row.composer_provider,
                composer_skill_hash=authority.row.composer_skill_hash,
                tool_arguments_hash=authority.row.tool_arguments_hash,
                raise_schema_argument_errors=True,
            ),
        )

    outcome = await dispatch_with_audit(
        recorder=recorder,
        audit=audit,
        do_dispatch=execute_exact,
        version_after_provider=lambda result: result.updated_state.version,
        arg_error_payload_factory=lambda exc: {
            "error_class": "ToolArgumentError",
            "error_code": exc.code or "argument_error",
        },
    )
    result = outcome.result
    if type(result) is not ToolResult:
        raise AuditIntegrityError("pipeline executor returned a non-ToolResult")
    captured = recorder.invocations[before_count:]
    if len(captured) != 1:
        raise AuditIntegrityError("pipeline commit dispatch must produce exactly one audit invocation")
    invocation = captured[0]
    if invocation.tool_call_id != authority.row.tool_call_id or invocation.arguments_hash != authority.row.tool_arguments_hash:
        raise AuditIntegrityError("pipeline commit dispatch audit does not bind exact proposal arguments")
    executor_hash = composition_content_hash(result.updated_state)
    invocation = _bind_executor_content_hash(invocation, executor_content_hash=executor_hash)
    binding = PipelineDispatchAuditBinding.from_invocation(invocation)
    if not result.success or not result.validation.is_valid:
        raise PipelineCommitError(
            "pipeline proposal failed current executor validation",
            code="VALIDATION_FAILED",
            invocation=invocation,
            dispatch=binding,
        )
    if candidate_hash != executor_hash:
        raise PipelineCommitMismatchError(invocation, binding)
    return PreparedPipelineCommit(
        result=result,
        candidate_content_hash=candidate_hash,
        executor_content_hash=executor_hash,
        invocation=invocation,
        dispatch=binding,
    )


__all__ = [
    "PipelineCommitConfig",
    "PipelineCommitError",
    "PipelineCommitMismatchError",
    "PipelineDispatchAuditBinding",
    "PreparedPipelineCommit",
    "RecoveredPipelineCommit",
    "prepare_pipeline_proposal_commit",
]
