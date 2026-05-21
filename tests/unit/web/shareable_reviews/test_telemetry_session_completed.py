"""Phase 8 Sub-task 7c (telemetry-backfill: phase-6) — counter emit
at ``ShareableReviewService.mark_ready_for_review``.

Tests that ``composer.session.completed_total`` increments by exactly
one with ``completion_verb="mark_ready_for_review"`` when the audit
row commits, and stays at zero when the audit insert raises (audit
primacy — the helper sits AFTER ``engine.begin()`` so the rollback
case can be observed structurally).

Q10 isolation discipline: every test builds its own
``build_sessions_telemetry()`` container so ``observed_value`` reads
are not cross-contaminated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import InvalidRequestError, OperationalError

from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.audit_readiness.models import AuditReadinessSnapshot, ReadinessRow
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    composer_completion_events_table,
    composition_states_table,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value
from elspeth.web.shareable_reviews.service import ShareableReviewService
from elspeth.web.shareable_reviews.signer import ShareTokenSigner

_VALID_SIGNING_KEY = b"k" * 32


# ── Fixtures (mirror test_service.py) ────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StateRecord:
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


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
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
def session_engine_with_row(engine, session_record: _SessionRecord, state_record: _StateRecord):  # type: ignore[no-untyped-def]
    """Insert the parent session + composition_state rows so the FK on
    ``composer_completion_events`` resolves at audit-insert time.
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
        readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
        semantic_contracts=[],
    )


def _readiness_snapshot(session_id: UUID) -> AuditReadinessSnapshot:
    def _row(row_id: str) -> ReadinessRow:
        return ReadinessRow(
            id=row_id,  # type: ignore[arg-type]
            label=row_id,
            status="ok",  # type: ignore[arg-type]
            summary="ok",
            detail=None,
            component_ids=(),
        )

    return AuditReadinessSnapshot(
        session_id=str(session_id),
        composition_version=3,
        checked_at=datetime.now(UTC),
        rows=(
            _row("validation"),
            _row("plugin_trust"),
            _row("provenance"),
            _row("retention"),
            _row("llm_interpretations"),
            _row("secrets"),
        ),
        validation_result=_ok_validation(),
    )


def _build_service_with_fresh_telemetry(  # type: ignore[no-untyped-def]
    *,
    engine,
    payload_store: FilesystemPayloadStore,
    signer: ShareTokenSigner,
    session_record: _SessionRecord,
    state_record: _StateRecord,
    readiness: AuditReadinessSnapshot,
):
    """Build a service with a freshly-constructed telemetry container.

    Returns the service paired with its telemetry container so tests
    can observe the counter via ``observed_value`` without reaching
    through internal attributes (the helper still passes the container
    in via the constructor — this just keeps the reference handy at
    test scope).
    """
    session_service = MagicMock()
    session_service.get_current_state = AsyncMock(return_value=state_record)
    session_service.get_session = AsyncMock(return_value=session_record)

    execution_service = MagicMock()
    execution_service.validate = AsyncMock(return_value=_ok_validation())
    execution_service.validate_state = AsyncMock(return_value=_ok_validation())

    readiness_service = MagicMock()
    readiness_service.compute_snapshot = AsyncMock(return_value=readiness)

    settings = MagicMock()
    settings.shareable_link_lifetime_seconds = 30 * 24 * 3600

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
    return service, telemetry


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_ready_for_review_emits_completion_counter(
    session_engine_with_row,  # type: ignore[no-untyped-def]
    payload_store: FilesystemPayloadStore,
    signer: ShareTokenSigner,
    session_record: _SessionRecord,
    state_record: _StateRecord,
) -> None:
    """Happy path: a successful ``mark_ready_for_review`` increments
    ``composer.session.completed_total`` exactly once with
    ``completion_verb="mark_ready_for_review"``, AND the audit row
    is present in ``composer_completion_events_table`` (audit primacy
    — the counter reflects an actual committed audit row).
    """
    snapshot = _readiness_snapshot(session_record.id)
    service, telemetry = _build_service_with_fresh_telemetry(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        readiness=snapshot,
    )

    # Baseline assertion before the call: counter starts at zero.
    assert observed_value(telemetry.session_completed_total) == 0

    response = await service.mark_ready_for_review(
        session_id=session_record.id,
        user_id=session_record.user_id,
    )
    assert response.token

    # Counter incremented exactly once with the DB-vocabulary verb.
    assert observed_value(telemetry.session_completed_total) == 1
    # The container fake records call attributes — assert the verb
    # attribute matches the audit-row event_type, NOT the UI vocab.
    from elspeth.web.sessions.telemetry import _FakeCounter

    counter = telemetry.session_completed_total
    assert isinstance(counter, _FakeCounter)
    assert counter.calls == [
        (1, {"completion_verb": "mark_ready_for_review"}, None),
    ]

    # Audit row is committed and carries the matching event_type. The
    # counter aggregates over THIS row (superset rule).
    with session_engine_with_row.connect() as conn:
        rows = conn.execute(
            select(composer_completion_events_table).where(
                composer_completion_events_table.c.session_id == str(session_record.id),
            )
        ).all()
    assert len(rows) == 1
    assert rows[0].event_type == "mark_ready_for_review"


@pytest.mark.asyncio
async def test_mark_ready_for_review_audit_failure_does_not_emit_counter(
    session_engine_with_row,  # type: ignore[no-untyped-def]
    payload_store: FilesystemPayloadStore,
    signer: ShareTokenSigner,
    session_record: _SessionRecord,
    state_record: _StateRecord,
) -> None:
    """Audit primacy regression: if the audit INSERT raises (we force
    this by disposing the engine before the call so ``engine.begin()``
    fails), the counter MUST stay at zero. The helper is placed
    structurally AFTER the ``engine.begin()`` block, so an exception
    out of the block skips the helper — that's the test surface for
    "counter reflects committed audit rows only".
    """
    snapshot = _readiness_snapshot(session_record.id)
    service, telemetry = _build_service_with_fresh_telemetry(
        engine=session_engine_with_row,
        payload_store=payload_store,
        signer=signer,
        session_record=session_record,
        state_record=state_record,
        readiness=snapshot,
    )

    # Dispose the engine — subsequent ``engine.begin()`` calls fail
    # because the underlying pool is closed. The audit INSERT raises,
    # the ``with`` block re-raises out of mark_ready_for_review, and
    # the helper after the block never executes.
    session_engine_with_row.dispose(close=True)

    assert observed_value(telemetry.session_completed_total) == 0

    # Narrowed from bare Exception to the specific SQLAlchemy classes a
    # post-dispose engine raises.  ``InvalidRequestError`` is the typical
    # raise from a closed pool on 2.x; ``OperationalError`` is the SQLite
    # connection-closed variant when the underlying connection is gone.
    # Both are stable across the 1.4/2.0 boundary.  Bare Exception was
    # hiding regressions where the engine silently succeeded — see
    # advisory commentary in S3 of the Phase 8 PR review.
    with pytest.raises((InvalidRequestError, OperationalError)):
        await service.mark_ready_for_review(
            session_id=session_record.id,
            user_id=session_record.user_id,
        )

    # Counter unchanged: audit primacy is structurally enforced by the
    # helper's placement AFTER engine.begin().
    assert observed_value(telemetry.session_completed_total) == 0
