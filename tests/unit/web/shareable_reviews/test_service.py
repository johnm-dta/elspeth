"""Tests for ``ShareableReviewService`` — business logic for the
shareable-reviews capability.

Phase 6A Task 5 (UX redesign 2026-05).

Service contract (see plan §"Task 5"):

* ``mark_ready_for_review`` — build snapshot, compute payload_digest,
  INSERT audit row (audit-first), write blob, sign token, return.
* ``get_shareable_link`` — re-mint a fresh token over the current snapshot;
  content-addressing makes this idempotent at the blob level.
* ``resolve_token`` — verify token, read blob by digest, return the
  frozen-at-mark-time snapshot. Does NOT call the readiness service.

Audit-first ordering is load-bearing: if the audit insert fails, no blob
is ever written and no token is returned. If the blob write fails after
the audit insert, the audit row stands as honest evidence of the attempt;
no token is returned.

Frozen-at-mark-time discipline: the ``audit_readiness`` field is computed
once at ``mark_ready_for_review`` time using the OWNER's user_id and
embedded in the snapshot blob. ``resolve_token`` reads it back from the
blob; it never re-calls ``ReadinessService.compute_snapshot()``. This
eliminates the reviewer-vs-owner permission question and ensures
content-addressing covers the readiness fingerprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import yaml
from sqlalchemy import select, text

from elspeth.contracts.payload_store import PayloadNotFoundError
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.audit_readiness.models import AuditReadinessSnapshot, ReadinessRow
from elspeth.web.execution.schemas import ValidationError, ValidationReadiness, ValidationResult
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    composer_completion_events_table,
    composition_states_table,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from elspeth.web.shareable_reviews.models import CompositionStateResponse
from elspeth.web.shareable_reviews.service import (
    CompositionNotRunnableError,
    ShareableReviewService,
)
from elspeth.web.shareable_reviews.signer import InvalidToken, ShareTokenSigner

_VALID_SIGNING_KEY = b"k" * 32


def _ready_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _blocked_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[])


def test_composition_snapshot_accepts_named_sources_payload() -> None:
    snapshot = CompositionStateResponse.model_validate(
        {
            "version": 1,
            "metadata": {"name": "legacy", "description": ""},
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "output",
                    "options": {"schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                }
            },
            "nodes": [],
            "edges": [],
            "outputs": [
                {
                    "name": "output",
                    "plugin": "json",
                    "options": {"path": "out.jsonl"},
                    "on_write_failure": "discard",
                }
            ],
        }
    )

    assert tuple(snapshot.sources) == ("primary",)
    assert snapshot.sources["primary"].plugin == "csv"


# ── Minimal record shims ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StateRecord:
    """Subset of CompositionStateRecord the service needs.

    The real CompositionStateRecord has many more fields; this shim carries
    only what ``state_from_record``-equivalent paths and the service's
    blob-normalisation path touch.
    """

    id: UUID
    session_id: UUID
    version: int
    metadata_: Any
    nodes: Any
    edges: Any
    outputs: Any
    source: Any
    sources: Any
    composer_meta: Any
    created_at: datetime


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    id: UUID
    user_id: str


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


@pytest.fixture
def payload_store(tmp_path: Path) -> FilesystemPayloadStore:
    return FilesystemPayloadStore(tmp_path / "blobs")


@pytest.fixture
def signer() -> ShareTokenSigner:
    return ShareTokenSigner(_VALID_SIGNING_KEY)


@pytest.fixture
def session_id() -> UUID:
    return uuid4()


@pytest.fixture
def state_id() -> UUID:
    return uuid4()


@pytest.fixture
def session_record(session_id: UUID) -> _SessionRecord:
    return _SessionRecord(id=session_id, user_id="alice")


@pytest.fixture
def state_record(session_id: UUID, state_id: UUID) -> _StateRecord:
    return _StateRecord(
        id=state_id,
        session_id=session_id,
        version=3,
        metadata_=MappingProxyType({"name": "Demo Pipeline", "description": ""}),
        nodes=(),
        edges=(),
        outputs=(),
        source=None,
        sources=None,
        composer_meta=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def session_engine_with_row(engine, session_record: _SessionRecord, state_record: _StateRecord):
    """Insert parent sessions + composition_states rows so FK constraints on
    composer_completion_events resolve.
    """
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=str(session_record.id),
                user_id=session_record.user_id,
                auth_provider_type="local",
                title="t",
                trust_mode="auto_commit",
                density_default="high",
                created_at=now,
                updated_at=now,
                interpretation_review_disabled=False,
            )
        )
        conn.execute(
            composition_states_table.insert().values(
                id=str(state_record.id),
                session_id=str(session_record.id),
                version=state_record.version,
                source=None,
                sources=None,
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={"name": "Demo Pipeline", "description": ""},
                is_valid=True,
                validation_errors=None,
                composer_meta=None,
                created_at=now,
                derived_from_state_id=None,
                provenance="tool_call",
            )
        )
    return engine


def _ok_validation() -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=_ready_readiness(),
        semantic_contracts=[],
    )


def _broken_validation() -> ValidationResult:
    return ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id="node1",
                component_type="transform",
                message="boom",
                suggestion=None,
                error_code=None,
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )


def _readiness_snapshot(
    session_id: UUID,
    version: int = 3,
    *,
    error_row: bool = False,
    warning_row: bool = False,
) -> AuditReadinessSnapshot:
    """Build a minimal valid AuditReadinessSnapshot for tests."""

    def _row(row_id: str, status: str = "ok", summary: str = "ok") -> ReadinessRow:
        return ReadinessRow(
            id=row_id,  # type: ignore[arg-type]
            label=row_id,
            status=status,  # type: ignore[arg-type]
            summary=summary,
            detail=None,
            component_ids=(),
        )

    validation_status = "ok"
    if error_row:
        validation_status = "error"
    elif warning_row:
        validation_status = "warning"

    return AuditReadinessSnapshot(
        session_id=str(session_id),
        composition_version=version,
        checked_at=datetime.now(UTC),
        rows=(
            _row("validation", status=validation_status, summary="check"),
            _row("plugin_trust"),
            _row("provenance"),
            _row("retention"),
            _row("llm_interpretations"),
            _row("secrets"),
        ),
        validation_result=_ok_validation(),
    )


def _build_service(
    *,
    engine,
    payload_store: FilesystemPayloadStore,
    signer: ShareTokenSigner,
    session_record: _SessionRecord,
    state_record: _StateRecord,
    validation: ValidationResult,
    readiness: AuditReadinessSnapshot,
) -> tuple[ShareableReviewService, MagicMock, MagicMock, MagicMock]:
    session_service = MagicMock()
    session_service.get_current_state = AsyncMock(return_value=state_record)
    session_service.get_session = AsyncMock(return_value=session_record)

    execution_service = MagicMock()
    execution_service.validate = AsyncMock(return_value=validation)
    execution_service.validate_state = AsyncMock(return_value=validation)

    readiness_service = MagicMock()
    readiness_service.compute_snapshot = AsyncMock(return_value=readiness)

    settings = MagicMock()
    settings.shareable_link_lifetime_seconds = 30 * 24 * 3600

    # Phase 8 Sub-task 7c — the constructor now requires a telemetry
    # container. Build a fresh fake-counter container per service
    # instance (Q10 isolation per the Phase 8 telemetry test discipline).
    # Tests asserting on the counter retrieve it via
    # ``service._telemetry`` (the test file in
    # ``test_telemetry_session_completed.py`` exercises that path
    # explicitly).
    telemetry = build_sessions_telemetry()

    service = ShareableReviewService(
        session_service=session_service,
        execution_service=execution_service,
        readiness_service=readiness_service,
        signer=signer,
        settings=settings,
        sessions_db_engine=engine,
        payload_store=payload_store,
        telemetry=telemetry,
    )
    return service, session_service, execution_service, readiness_service


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_ready_for_review_happy_path(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    snapshot = _readiness_snapshot(session_record.id)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    assert response.token
    assert response.payload_digest.startswith("sha256:")
    # Blob is in the payload store (hex digest without prefix).
    digest_hex = response.payload_digest.removeprefix("sha256:")
    assert payload_store.exists(digest_hex)
    # Audit row landed.
    with session_engine_with_row.connect() as conn:
        rows = conn.execute(
            select(composer_completion_events_table).where(composer_completion_events_table.c.session_id == str(session_record.id))
        ).all()
    assert len(rows) == 1
    assert rows[0].event_type == "mark_ready_for_review"
    assert rows[0].actor == session_record.user_id
    assert rows[0].payload_digest == response.payload_digest
    # Token verifies through the same signer.
    payload = signer.verify(response.token)
    assert payload.session_id == session_record.id
    assert payload.payload_digest == response.payload_digest


@pytest.mark.asyncio
async def test_mark_ready_for_review_yaml_strips_blob_bound_source_storage_path(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    storage_path = "/data/blobs/session/98b1357d_contact_form_submissions.csv"
    blob_id = "98b1357d-5aab-4fb3-85b4-5ad643912e84"
    blob_state_record = replace(
        state_record,
        source={
            "plugin": "csv",
            "on_success": "main",
            "options": {
                "path": storage_path,
                "blob_ref": blob_id,
                "mode": "bind_source",
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "quarantine",
        },
        outputs=[
            {
                "name": "main",
                "plugin": "csv",
                "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            }
        ],
    )
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=blob_state_record,
        validation=_ok_validation(),
        readiness=_readiness_snapshot(session_record.id),
    )

    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)

    payload = json.loads(payload_store.retrieve(response.payload_digest.removeprefix("sha256:")).decode("utf-8"))
    assert storage_path not in payload["yaml"]
    options = yaml.safe_load(payload["yaml"])["sources"]["source"]["options"]
    assert "path" not in options
    assert "blob_ref" not in options
    assert "mode" not in options
    assert options["schema"] == {"mode": "observed"}


@pytest.mark.asyncio
async def test_mark_ready_for_review_fails_validation(session_engine_with_row, payload_store, signer, session_record, state_record):
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_broken_validation(),
        readiness=_readiness_snapshot(session_record.id),
    )
    with pytest.raises(CompositionNotRunnableError):
        await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    # No audit row was written, no blob was stored.
    with session_engine_with_row.connect() as conn:
        rows = conn.execute(select(composer_completion_events_table)).all()
    assert rows == []


@pytest.mark.asyncio
async def test_mark_ready_for_review_blocks_error_readiness_row(
    session_engine_with_row, payload_store, signer, session_record, state_record
):
    """Sharing a state with status='error' on any readiness row is share-theatre."""
    snapshot = _readiness_snapshot(session_record.id, error_row=True)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    with pytest.raises(CompositionNotRunnableError):
        await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)


@pytest.mark.asyncio
async def test_mark_ready_for_review_allows_warning_readiness_row(
    session_engine_with_row, payload_store, signer, session_record, state_record
):
    """status='warning' (e.g. pending llm_interpretations) is NOT a blocker."""
    snapshot = _readiness_snapshot(session_record.id, warning_row=True)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    assert response.token


@pytest.mark.asyncio
async def test_mark_ready_for_review_audit_first_ordering(
    session_engine_with_row, payload_store, signer, session_record, state_record, monkeypatch
):
    """If the audit insert fails, no blob is ever written.

    Audit-first ordering: the sequence is build → digest → audit INSERT →
    blob write → sign token. A failing audit must short-circuit before any
    file is written.
    """
    snapshot = _readiness_snapshot(session_record.id)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )

    # Track payload_store.store invocations so we can prove it was not called.
    original_store = payload_store.store
    store_calls: list[bytes] = []

    def tracking_store(content: bytes) -> str:
        store_calls.append(content)
        return original_store(content)

    monkeypatch.setattr(payload_store, "store", tracking_store)

    # Drop the sessions row to provoke an audit FK insert failure.
    with session_engine_with_row.begin() as conn:
        # The CHILD table has BEFORE DELETE ABORT trigger — sessions table
        # itself is not append-only, so we can manually break the FK by
        # inserting an event with a missing session_id. Easier: drop the
        # sessions row entirely so the FK fails on insert.
        conn.execute(text("DELETE FROM sessions WHERE id = :id"), {"id": str(session_record.id)})

    with pytest.raises(Exception):  # noqa: B017 — any IntegrityError variant fails the request
        await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    # CRITICAL: no blob was written.
    assert store_calls == [], "audit insert must precede blob write — blob should not exist when audit fails"


@pytest.mark.asyncio
async def test_get_shareable_link_requires_mark_ready_for_current_snapshot(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
    monkeypatch,
):
    """Re-minting before mark-ready must not create an unaudited share artifact."""
    snapshot = _readiness_snapshot(session_record.id)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )

    store_calls: list[bytes] = []

    def tracking_store(content: bytes) -> str:
        store_calls.append(content)
        return "unused"

    monkeypatch.setattr(payload_store, "store", tracking_store)

    with pytest.raises(CompositionNotRunnableError) as exc_info:
        await service.get_shareable_link(session_id=session_record.id, user_id=session_record.user_id)

    assert exc_info.value.reason == "completion_event_missing"
    assert store_calls == []


@pytest.mark.asyncio
async def test_get_shareable_link_rejects_state_drift_even_when_digest_matches(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    """The prior mark must match both current state_id and payload_digest."""
    snapshot = _readiness_snapshot(session_record.id)
    service, session_service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )

    await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)

    drifted_state = replace(state_record, id=uuid4())
    session_service.get_current_state = AsyncMock(return_value=drifted_state)

    with pytest.raises(CompositionNotRunnableError) as exc_info:
        await service.get_shareable_link(session_id=session_record.id, user_id=session_record.user_id)

    assert exc_info.value.reason == "completion_event_missing"


@pytest.mark.asyncio
async def test_get_shareable_link_requires_existing_mark_ready_blob(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    """An audit attempt row without the blob is not a successful share mark."""
    snapshot = _readiness_snapshot(session_record.id)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    payload_store.delete(response.payload_digest.removeprefix("sha256:"))

    with pytest.raises(CompositionNotRunnableError) as exc_info:
        await service.get_shareable_link(session_id=session_record.id, user_id=session_record.user_id)

    assert exc_info.value.reason == "not_marked_ready"


@pytest.mark.asyncio
async def test_get_shareable_link_remints_with_stable_digest(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    """Two get_shareable_link calls on an unchanged state yield identical digests
    but different token strings (different nonce each call). Re-minting writes
    no new audit rows: the single mark_ready_for_review row remains the only
    share decision on record."""
    snapshot = _readiness_snapshot(session_record.id)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    marked = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    r1 = await service.get_shareable_link(session_id=session_record.id, user_id=session_record.user_id)
    r2 = await service.get_shareable_link(session_id=session_record.id, user_id=session_record.user_id)
    assert r1.payload_digest == marked.payload_digest
    assert r1.payload_digest == r2.payload_digest
    assert r1.token != r2.token

    with session_engine_with_row.connect() as conn:
        rows = conn.execute(
            select(
                composer_completion_events_table.c.payload_digest,
                composer_completion_events_table.c.expires_at,
            )
            .where(composer_completion_events_table.c.session_id == str(session_record.id))
            .where(composer_completion_events_table.c.event_type == "mark_ready_for_review")
            .order_by(composer_completion_events_table.c.created_at)
        ).all()

    assert [row.payload_digest for row in rows] == [marked.payload_digest]
    assert rows[0].payload_digest == r1.payload_digest
    assert rows[0].payload_digest == r2.payload_digest
    assert rows[0].expires_at.replace(tzinfo=UTC) == marked.expires_at


@pytest.mark.asyncio
async def test_get_shareable_link_requires_mark_ready_event(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    service, _ss, execution_service, readiness_service = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=_readiness_snapshot(session_record.id),
    )

    with pytest.raises(CompositionNotRunnableError, match="mark this composition ready"):
        await service.get_shareable_link(session_id=session_record.id, user_id=session_record.user_id)

    execution_service.validate_state.assert_not_awaited()
    readiness_service.compute_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_mark_ready_for_review_rejects_readiness_snapshot_drift(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    service, _ss, _es, _rs = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=_readiness_snapshot(session_record.id, version=state_record.version + 1),
    )

    with pytest.raises(CompositionNotRunnableError, match="composition changed"):
        await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)


@pytest.mark.asyncio
async def test_resolve_token_returns_frozen_snapshot(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    """resolve_token returns the mark-time audit_readiness even if the live state shifts."""
    mark_time_snapshot = _readiness_snapshot(session_record.id, version=3)
    service, _ss, _es, readiness_service = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=mark_time_snapshot,
    )
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    # Mutate the readiness mock so a re-fetch would return a different snapshot.
    later_snapshot = _readiness_snapshot(session_record.id, version=99)
    readiness_service.compute_snapshot = AsyncMock(return_value=later_snapshot)

    resolved = await service.resolve_token(token=response.token, requesting_user_id="bob")
    # Frozen-at-mark-time: composition_version reflects the mark-time view, not the later one.
    assert resolved.audit_readiness.composition_version == 3
    # The readiness service was NOT called during resolve.
    assert readiness_service.compute_snapshot.call_count == 0


@pytest.mark.asyncio
async def test_resolve_token_rejects_tampered_token(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=_readiness_snapshot(session_record.id),
    )
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    tampered = response.token[:-2] + ("aa" if response.token[-2:] != "aa" else "bb")
    with pytest.raises(InvalidToken):
        await service.resolve_token(token=tampered, requesting_user_id="bob")


@pytest.mark.asyncio
async def test_resolve_token_rejects_expired_token(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    """Override the lifetime to negative so the minted token is born expired."""
    snapshot = _readiness_snapshot(session_record.id)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    service._settings.shareable_link_lifetime_seconds = -1  # type: ignore[attr-defined]
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    with pytest.raises(InvalidToken, match="expired"):
        await service.resolve_token(token=response.token, requesting_user_id="bob")


@pytest.mark.asyncio
async def test_resolve_token_blob_expired_raises_not_found(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    """Token verifies but payload_store has expired the blob → 404 path."""
    snapshot = _readiness_snapshot(session_record.id)
    service, *_ = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    # Delete the blob.
    digest_hex = response.payload_digest.removeprefix("sha256:")
    payload_store.delete(digest_hex)
    with pytest.raises(PayloadNotFoundError):
        await service.resolve_token(token=response.token, requesting_user_id="bob")


@pytest.mark.asyncio
async def test_resolve_token_does_not_call_readiness_service(
    session_engine_with_row,
    payload_store,
    signer,
    session_record,
    state_record,
):
    """Frozen-at-mark-time discipline: resolve never calls compute_snapshot."""
    snapshot = _readiness_snapshot(session_record.id)
    service, _ss, _es, readiness_service = _build_service(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        validation=_ok_validation(),
        readiness=snapshot,
    )
    response = await service.mark_ready_for_review(session_id=session_record.id, user_id=session_record.user_id)
    readiness_service.compute_snapshot.reset_mock()
    await service.resolve_token(token=response.token, requesting_user_id="bob")
    assert readiness_service.compute_snapshot.call_count == 0
