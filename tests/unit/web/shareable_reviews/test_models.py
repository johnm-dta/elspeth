"""Tests for shareable_reviews response models.

Phase 6A Task 4 (UX redesign 2026-05). All three response models are
``_StrictResponse``-derived: ``strict=True, extra="forbid"``. The plan
re-uses the Phase 2 ``AuditReadinessSnapshot`` model verbatim inside
``SharedInspectResponse`` so the shared inspect view shows the same
six-row readiness panel the owner sees.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from elspeth.web.audit_readiness.models import AuditReadinessSnapshot, ReadinessRow
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult
from elspeth.web.shareable_reviews.models import (
    MarkReadyForReviewResponse,
    ShareableLinkResponse,
    SharedInspectResponse,
)


def _make_composition_snapshot() -> dict[str, object]:
    """Build a minimal valid composition_snapshot wire shape.

    Mirrors the dict shape produced by ``CompositionState.to_dict()`` —
    version, metadata, sources, nodes, edges, outputs. With the FIX-A
    tightening of SharedInspectResponse (Plan 19a:891-892) the
    composition_snapshot field is a strict Pydantic model and partial
    dicts are rejected at construction.
    """
    return {
        "version": 1,
        "metadata": {"name": "Demo", "description": ""},
        "sources": {},
        "nodes": [],
        "edges": [],
        "outputs": [],
    }


def _make_audit_readiness_snapshot() -> AuditReadinessSnapshot:
    """Build a minimal valid AuditReadinessSnapshot for response-model tests.

    Phase 2 contract requires all six closed-enum row ids to be present.
    """

    def _row(row_id: str) -> ReadinessRow:
        return ReadinessRow(
            id=row_id,  # type: ignore[arg-type]
            label=row_id,
            status="ok",
            summary="ok",
            detail=None,
            component_ids=(),
        )

    validation_result = ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
        semantic_contracts=[],
    )
    return AuditReadinessSnapshot(
        session_id="s-1",
        composition_version=1,
        checked_at=datetime.now(UTC),
        rows=tuple(
            _row(row_id)
            for row_id in (
                "validation",
                "plugin_trust",
                "provenance",
                "retention",
                "llm_interpretations",
                "secrets",
            )
        ),
        validation_result=validation_result,
    )


def test_mark_ready_response_strict() -> None:
    resp = MarkReadyForReviewResponse(
        token="abc123",
        share_url="https://example.com/shared/abc123",
        expires_at=datetime.now(UTC),
        payload_digest="sha256:" + ("ab" * 32),
    )
    assert resp.token == "abc123"
    assert resp.payload_digest.startswith("sha256:")


def test_mark_ready_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        MarkReadyForReviewResponse(
            token="abc",
            share_url="https://x/",
            expires_at=datetime.now(UTC),
            payload_digest="sha256:" + ("ab" * 32),
            unexpected="field",  # type: ignore[call-arg]
        )


def test_mark_ready_rejects_str_for_datetime() -> None:
    """strict=True forbids str→datetime coercion."""
    with pytest.raises(ValidationError):
        MarkReadyForReviewResponse(
            token="abc",
            share_url="https://x/",
            expires_at="2026-01-01T00:00:00+00:00",  # type: ignore[arg-type]
            payload_digest="sha256:" + ("ab" * 32),
        )


def test_shareable_link_response_strict() -> None:
    resp = ShareableLinkResponse(
        token="abc",
        share_url="https://x/abc",
        expires_at=datetime.now(UTC),
        state_id=str(uuid4()),
        payload_digest="sha256:" + ("cd" * 32),
    )
    assert resp.payload_digest.startswith("sha256:")


def test_shareable_link_response_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        ShareableLinkResponse(
            token="abc",
            share_url="https://x/abc",
            expires_at=datetime.now(UTC),
            state_id=str(uuid4()),
            payload_digest="sha256:" + ("cd" * 32),
            extra="boom",  # type: ignore[call-arg]
        )


def test_shared_inspect_response_carries_audit_readiness() -> None:
    """SharedInspectResponse must include audit_readiness (consumed by 19b Task 8)."""
    snapshot = _make_audit_readiness_snapshot()
    resp = SharedInspectResponse(
        session_id=str(uuid4()),
        state_id=str(uuid4()),
        pipeline_metadata={"name": "Demo", "description": ""},
        composition_snapshot=_make_composition_snapshot(),
        yaml="version: 1\n",
        audit_readiness=snapshot,
        created_by_user_id="user-1",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
    )
    assert resp.audit_readiness is snapshot
    assert resp.yaml == "version: 1\n"


def test_shared_inspect_response_accepts_plural_sources_snapshot() -> None:
    """Shareable-review snapshots accept only CompositionState's canonical sources map."""
    snapshot = _make_audit_readiness_snapshot()
    composition = _make_composition_snapshot()
    source = {
        "plugin": "csv",
        "on_success": "normalize",
        "options": {"schema": {"mode": "observed"}},
        "on_validation_failure": "discard",
    }
    composition["sources"] = {"source": source}

    resp = SharedInspectResponse(
        session_id=str(uuid4()),
        state_id=str(uuid4()),
        pipeline_metadata={"name": "Demo", "description": ""},
        composition_snapshot=composition,
        yaml="version: 1\n",
        audit_readiness=snapshot,
        created_by_user_id="user-1",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
    )

    assert resp.composition_snapshot.sources["source"].plugin == "csv"


def test_shared_inspect_response_rejects_extra_field() -> None:
    snapshot = _make_audit_readiness_snapshot()
    with pytest.raises(ValidationError):
        SharedInspectResponse(
            session_id=str(uuid4()),
            state_id=str(uuid4()),
            pipeline_metadata={"name": "Demo", "description": ""},
            composition_snapshot=_make_composition_snapshot(),
            yaml="version: 1\n",
            audit_readiness=snapshot,
            created_by_user_id="user-1",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
            something_extra=True,  # type: ignore[call-arg]
        )


def test_shared_inspect_response_requires_audit_readiness() -> None:
    """Omitting audit_readiness must fail (it's not Optional)."""
    with pytest.raises(ValidationError, match="audit_readiness"):
        SharedInspectResponse(  # type: ignore[call-arg]
            session_id=str(uuid4()),
            state_id=str(uuid4()),
            pipeline_metadata={"name": "Demo", "description": ""},
            composition_snapshot=_make_composition_snapshot(),
            yaml="version: 1\n",
            created_by_user_id="user-1",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
        )


# ── FIX-A: drift-crashes-at-construction tests (Plan 19a:891-892) ─────────


def test_shared_inspect_response_rejects_extra_pipeline_metadata_key() -> None:
    """An unknown key inside pipeline_metadata must crash at construction.

    Plan 19a:64 — "drift crashes at construction" — depends on the
    response model being a strict Pydantic mirror of PipelineMetadata,
    not a free-form dict. An unknown key (producer drift) must crash so
    audit-shape regressions are caught at the wire boundary, not silently
    accepted.
    """
    snapshot = _make_audit_readiness_snapshot()
    with pytest.raises(ValidationError):
        SharedInspectResponse(
            session_id=str(uuid4()),
            state_id=str(uuid4()),
            pipeline_metadata={"name": "Demo", "description": "", "unknown_key": True},
            composition_snapshot=_make_composition_snapshot(),
            yaml="version: 1\n",
            audit_readiness=snapshot,
            created_by_user_id="user-1",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
        )


def test_shared_inspect_response_rejects_extra_composition_snapshot_key() -> None:
    """An unknown top-level key inside composition_snapshot must crash."""
    snapshot = _make_audit_readiness_snapshot()
    bad_snapshot = _make_composition_snapshot()
    bad_snapshot["unknown_top_level_key"] = "boom"
    with pytest.raises(ValidationError):
        SharedInspectResponse(
            session_id=str(uuid4()),
            state_id=str(uuid4()),
            pipeline_metadata={"name": "Demo", "description": ""},
            composition_snapshot=bad_snapshot,
            yaml="version: 1\n",
            audit_readiness=snapshot,
            created_by_user_id="user-1",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
        )


def test_shared_inspect_response_rejects_wrong_pipeline_metadata_field_type() -> None:
    """A wrong-type value inside pipeline_metadata must crash.

    strict=True forbids int→str coercion. ``name`` is declared as str
    on the response mirror; passing ``123`` must raise rather than
    silently coercing to ``"123"``.
    """
    snapshot = _make_audit_readiness_snapshot()
    with pytest.raises(ValidationError):
        SharedInspectResponse(
            session_id=str(uuid4()),
            state_id=str(uuid4()),
            pipeline_metadata={"name": 123, "description": ""},
            composition_snapshot=_make_composition_snapshot(),
            yaml="version: 1\n",
            audit_readiness=snapshot,
            created_by_user_id="user-1",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
        )
