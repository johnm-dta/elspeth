"""Tests for audit-readiness Pydantic response models."""

from datetime import UTC, datetime
from typing import get_args

import pytest
from pydantic import ValidationError

from elspeth.web.audit_readiness.models import (
    AuditReadinessExplain,
    AuditReadinessSnapshot,
    ReadinessRow,
    ReadinessRowId,
    ReadinessStatus,
)
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult

CHECKED_AT = datetime(2026, 5, 17, 3, 30, tzinfo=UTC)


def _row(row_id, status="ok"):
    return ReadinessRow(
        id=row_id,
        label=row_id,
        status=status,
        summary="x",
        detail=None,
        component_ids=(),
    )


def _validation_result():
    return ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
        semantic_contracts=[],
    )


def test_row_constructs_with_minimal_fields():
    row = _row("validation")
    assert row.status == "ok"


def test_row_rejects_unknown_id():
    with pytest.raises(ValidationError):
        ReadinessRow(id="kiosk", label="x", status="ok", summary="x", detail=None, component_ids=())


def test_row_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ReadinessRow(id="validation", label="x", status="purple", summary="x", detail=None, component_ids=())


def test_row_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ReadinessRow(id="validation", label="x", status="ok", summary="x", detail=None, component_ids=(), sneaky="oops")


def test_snapshot_emits_six_canonical_rows():
    rows = tuple(
        _row(r)
        for r in (
            "validation",
            "plugin_trust",
            "provenance",
            "retention",
            "llm_interpretations",
            "secrets",
        )
    )
    snap = AuditReadinessSnapshot(
        session_id="11111111-1111-1111-1111-111111111111",
        composition_version=1,
        checked_at=CHECKED_AT,
        rows=rows,
        validation_result=_validation_result(),
    )
    assert {row.id for row in snap.rows} == set(get_args(ReadinessRowId))


def test_snapshot_requires_checked_at():
    rows = tuple(
        _row(r)
        for r in (
            "validation",
            "plugin_trust",
            "provenance",
            "retention",
            "llm_interpretations",
            "secrets",
        )
    )
    with pytest.raises(ValidationError, match="checked_at"):
        AuditReadinessSnapshot(
            session_id="11111111-1111-1111-1111-111111111111",
            composition_version=1,
            rows=rows,
            validation_result=_validation_result(),
        )


def test_snapshot_accepts_utc_checked_at():
    rows = tuple(
        _row(r)
        for r in (
            "validation",
            "plugin_trust",
            "provenance",
            "retention",
            "llm_interpretations",
            "secrets",
        )
    )
    snap = AuditReadinessSnapshot(
        session_id="11111111-1111-1111-1111-111111111111",
        composition_version=1,
        checked_at=CHECKED_AT,
        rows=rows,
        validation_result=_validation_result(),
    )
    assert snap.checked_at == CHECKED_AT


def test_snapshot_rejects_duplicate_rows():
    rows = (_row("validation"), _row("validation"))
    with pytest.raises(ValidationError, match="duplicate"):
        AuditReadinessSnapshot(
            session_id="11111111-1111-1111-1111-111111111111",
            composition_version=1,
            checked_at=CHECKED_AT,
            rows=rows,
            validation_result=_validation_result(),
        )


def test_snapshot_rejects_missing_rows():
    rows = (_row("validation"),)
    with pytest.raises(ValidationError, match="missing"):
        AuditReadinessSnapshot(
            session_id="11111111-1111-1111-1111-111111111111",
            composition_version=1,
            checked_at=CHECKED_AT,
            rows=rows,
            validation_result=_validation_result(),
        )


def test_explain_constructs():
    ex = AuditReadinessExplain(
        session_id="11111111-1111-1111-1111-111111111111",
        composition_version=1,
        narrative="When you run this pipeline, ELSPETH will record:\n- foo",
    )
    assert "ELSPETH" in ex.narrative


def test_explain_rejects_empty_narrative():
    with pytest.raises(ValidationError):
        AuditReadinessExplain(
            session_id="11111111-1111-1111-1111-111111111111",
            composition_version=1,
            narrative="",
        )


def test_status_literal_closed_set():
    assert set(get_args(ReadinessStatus)) == {
        "ok",
        "warning",
        "error",
        "not_applicable",
    }
