"""Pydantic models for the audit-readiness endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, get_args

from pydantic import Field, model_validator

from elspeth.web.execution.schemas import ValidationResult, _StrictResponse

# Maps 1:1 to panel rows per docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md.
# Adding a row requires updating ReadinessService and Phase 2B's renderer.
ReadinessRowId = Literal[
    "validation",
    "plugin_trust",
    "provenance",
    "retention",
    "llm_interpretations",
    "secrets",
]

# Panel glyphs: ok→✓, warning→⚠, error→✗, not_applicable→—.
ReadinessStatus = Literal["ok", "warning", "error", "not_applicable"]

PluginPolicyReadinessRowId = Literal[
    "policy_compilation",
    "required_core",
    "local_capability_configuration",
    "live_health",
    "tutorial_profile",
    "tutorial_required_control_coverage",
]

_EXPECTED_PLUGIN_POLICY_ROW_IDS: frozenset[str] = frozenset(get_args(PluginPolicyReadinessRowId))

_EXPECTED_ROW_IDS: frozenset[str] = frozenset(get_args(ReadinessRowId))


class ReadinessRow(_StrictResponse):
    """One row in the audit-readiness panel."""

    id: ReadinessRowId
    label: str = Field(min_length=1)
    status: ReadinessStatus
    summary: str = Field(min_length=1)
    detail: str | None
    # component_ids let the frontend render jump-to-where links (Phase 2B).
    # Empty when the row is system-scoped (retention) or all-green.
    component_ids: tuple[str, ...]


class PluginPolicyReadinessRow(_StrictResponse):
    """One sanitized process/request plugin-policy readiness signal."""

    id: PluginPolicyReadinessRowId
    label: str = Field(min_length=1)
    status: ReadinessStatus
    summary: str = Field(min_length=1)
    detail: str | None


class PluginPolicyReadinessSnapshot(_StrictResponse):
    """The fixed policy/tutorial readiness matrix exposed to web surfaces."""

    rows: tuple[PluginPolicyReadinessRow, ...]
    tutorial_ready: bool

    @model_validator(mode="after")
    def _check_row_completeness(self) -> Self:
        ids = [row.id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate plugin-policy readiness row ids: {ids}")
        missing = _EXPECTED_PLUGIN_POLICY_ROW_IDS - set(ids)
        if missing:
            raise ValueError(f"plugin-policy readiness missing required rows: {sorted(missing)}")
        return self


class AuditReadinessSnapshot(_StrictResponse):
    """Aggregated payload for the audit-readiness panel."""

    session_id: str = Field(min_length=1)
    composition_version: int = Field(ge=1)
    checked_at: datetime
    rows: tuple[ReadinessRow, ...]
    # Raw validation output from the same state snapshot as the summary rows.
    # The frontend uses this for structured component attribution rather than
    # reconstructing a lossy ValidationResult from the validation row.
    validation_result: ValidationResult
    plugin_policy_readiness: PluginPolicyReadinessSnapshot | None = None

    @model_validator(mode="after")
    def _check_row_completeness(self) -> Self:
        # ReadinessRow.id is typed Literal[ReadinessRowId]; Pydantic rejects
        # any other value at row construction, so checking for "extra" ids
        # here would be unreachable. Only duplicate-id and missing-id remain
        # as real failure modes.
        ids = [row.id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate row ids in snapshot: {ids}")
        missing = _EXPECTED_ROW_IDS - set(ids)
        if missing:
            raise ValueError(f"snapshot missing required rows: {sorted(missing)}")
        return self


class AuditReadinessExplain(_StrictResponse):
    """Narrative form for the Explain detail view."""

    session_id: str = Field(min_length=1)
    composition_version: int = Field(ge=1)
    narrative: str = Field(min_length=1)


class SinkEffectAttemptDiagnostic(_StrictResponse):
    """One bounded, credential-free external-call witness."""

    attempt_id: str = Field(min_length=1)
    attempt_index: int = Field(ge=0)
    member_ordinal: int | None = Field(default=None, ge=0)
    generation: int = Field(ge=0)
    action: Literal["inspect", "commit", "reconcile"]
    call_kind: str = Field(min_length=1)
    request_hash: str = Field(min_length=64, max_length=64)
    state: Literal["intent", "returned", "response_lost", "error"]
    evidence_hash: str | None = Field(default=None, min_length=64, max_length=64)
    started_at: datetime
    completed_at: datetime | None
    latency_ms: float | None = Field(default=None, ge=0)


class SinkEffectRecoveryDiagnostic(_StrictResponse):
    """Web-safe operator view of a recoverable publication effect."""

    effect_id: str = Field(min_length=64, max_length=64)
    run_id: str = Field(min_length=1)
    sink_node_id: str = Field(min_length=1)
    state: Literal["reserved", "prepared", "in_flight", "finalized"]
    predecessor_effect_id: str | None
    lease_owner: str | None
    lease_generation: int = Field(ge=0)
    lease_expires_at: datetime | None
    reconcile_kind: Literal["not_applied", "applied_with_exact_descriptor", "unknown"] | None
    result_descriptor_hash: str | None = Field(default=None, min_length=64, max_length=64)
    publication_performed: bool | None
    publication_evidence_kind: str | None
    member_progress: dict[str, int]
    response_lost_attempts: int = Field(ge=0)
    attempts: tuple[SinkEffectAttemptDiagnostic, ...]
    operator_guidance: str = Field(min_length=1)
