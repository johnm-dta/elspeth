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
