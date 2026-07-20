from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import UUID

import pytest
import structlog
from sqlalchemy import delete, insert, inspect, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.hashing import stable_hash
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    composition_proposals_table,
    guided_operation_admission_blocks_table,
    guided_operation_events_table,
    guided_operations_table,
    metadata,
    sessions_table,
)
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import SessionSchemaError, initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

SESSION_ID = "00000000-0000-4000-8000-000000000001"
OPERATION_ID = "00000000-0000-4000-8000-000000000002"
STATE_ID = "00000000-0000-4000-8000-000000000004"
PROPOSAL_ID = "00000000-0000-4000-8000-000000000008"
RESULT_SESSION_ID = "00000000-0000-4000-8000-000000000009"
REQUEST_HASH = "a" * 64
RESPONSE_HASH = "b" * 64
NOW = datetime(2026, 7, 18, tzinfo=UTC)


def _empty_failure_audit_cohort() -> dict[str, object]:
    authority: dict[str, object] = {
        "schema": "guided_failure_audit_cohort.v1",
        "count": 0,
        "rows": [],
    }
    return {**authority, "aggregate_digest": stable_hash(authority)}


@pytest.fixture
def engine():
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(sessions_table).values(
                id=SESSION_ID,
                user_id="schema-test",
                auth_provider_type="local",
                title="Guided operation",
                trust_mode="auto_commit",
                density_default="high",
                created_at=NOW,
                updated_at=NOW,
                interpretation_review_disabled=False,
            )
        )
        connection.execute(
            insert(sessions_table).values(
                id=RESULT_SESSION_ID,
                user_id="schema-test",
                auth_provider_type="local",
                title="Fork result",
                trust_mode="auto_commit",
                density_default="high",
                created_at=NOW,
                updated_at=NOW,
                interpretation_review_disabled=False,
            )
        )
        connection.execute(
            insert(composition_proposals_table).values(
                id=PROPOSAL_ID,
                session_id=SESSION_ID,
                tool_call_id="schema-call",
                tool_name="set_pipeline",
                status="pending",
                summary="Schema proposal",
                rationale="Schema proof",
                affects=[],
                arguments_json={},
                arguments_redacted_json={},
                created_at=NOW,
                updated_at=NOW,
            )
        )
    service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided-operation-schema"),
    )
    with patch("elspeth.web.sessions.service.uuid.uuid4", return_value=UUID(STATE_ID)):
        state = asyncio.run(
            service.save_composition_state(
                UUID(SESSION_ID),
                CompositionStateData(is_valid=False),
                provenance="session_seed",
            )
        )
    assert str(state.id) == STATE_ID
    return engine


def _operation(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": SESSION_ID,
        "operation_id": OPERATION_ID,
        "kind": "guided_start",
        "status": "in_progress",
        "request_hash": REQUEST_HASH,
        "lease_token": "lease-token",
        "lease_expires_at": NOW + timedelta(minutes=1),
        "attempt": 1,
        "originating_message_id": None,
        "proposal_id": None,
        "result_kind": None,
        "result_state_id": None,
        "result_session_id": None,
        "response_hash": None,
        "failure_code": None,
        "created_at": NOW,
        "updated_at": NOW,
        "settled_at": None,
    }
    values.update(overrides)
    return values


def _event(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": SESSION_ID,
        "operation_id": OPERATION_ID,
        "sequence": 1,
        "event_kind": "claimed",
        "actor": "worker-a",
        "attempt": 1,
        "prior_attempt": None,
        "lease_expires_at": NOW + timedelta(minutes=1),
        "request_hash": REQUEST_HASH,
        "failure_audit_cohort": None,
        "occurred_at": NOW,
    }
    values.update(overrides)
    return values


def _admission_block(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": SESSION_ID,
        "operation_id": OPERATION_ID,
        "kind": "guided_start",
        "failure_code": "request_cancelled",
        "actor": "reconciler",
        "created_at": NOW,
    }
    values.update(overrides)
    return values


def test_tables_and_composite_keys_are_current(engine) -> None:
    assert {
        "guided_operations",
        "guided_operation_events",
        "guided_operation_admission_blocks",
    } <= set(metadata.tables)
    inspector = inspect(engine)
    assert inspector.get_pk_constraint("guided_operations")["constrained_columns"] == ["session_id", "operation_id"]
    assert inspector.get_pk_constraint("guided_operation_events")["constrained_columns"] == [
        "session_id",
        "operation_id",
        "sequence",
    ]
    assert inspector.get_pk_constraint("guided_operation_admission_blocks")["constrained_columns"] == [
        "session_id",
        "operation_id",
    ]
    assert {column["name"] for column in inspector.get_columns("guided_operation_admission_blocks")} == {
        "session_id",
        "operation_id",
        "kind",
        "failure_code",
        "actor",
        "created_at",
    }
    assert "failure_audit_cohort" in {column["name"] for column in inspector.get_columns("guided_operation_events")}


@pytest.mark.parametrize(
    "event",
    [
        _event(event_kind="failed", lease_expires_at=None),
        _event(failure_audit_cohort=_empty_failure_audit_cohort()),
    ],
    ids=["failed-without-commitment", "non-failed-with-commitment"],
)
def test_failure_event_commitment_presence_is_exact(engine, event: dict[str, object]) -> None:
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation()))
    with pytest.raises(IntegrityError, match="ck_guided_operation_events_failure_audit_cohort"), engine.begin() as connection:
        connection.execute(insert(guided_operation_events_table).values(**event))


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation_id": ""},
        {"kind": "guided_chat"},
        {"failure_code": "operation_failed"},
        {"actor": ""},
    ],
    ids=["empty-operation-id", "wrong-kind", "wrong-failure-code", "empty-actor"],
)
def test_admission_block_has_a_closed_non_sensitive_shape(engine, overrides: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(guided_operation_admission_blocks_table).values(**_admission_block(**overrides)))


def test_admission_block_and_operation_cannot_coexist_in_either_insert_order(engine) -> None:
    with engine.begin() as connection:
        connection.execute(insert(guided_operation_admission_blocks_table).values(**_admission_block()))
    with pytest.raises(IntegrityError, match="admission"), engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation()))

    other_operation_id = "00000000-0000-4000-8000-000000000012"
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation(operation_id=other_operation_id)))
    with pytest.raises(IntegrityError, match="admission"), engine.begin() as connection:
        connection.execute(insert(guided_operation_admission_blocks_table).values(**_admission_block(operation_id=other_operation_id)))


def test_operation_cannot_be_rekeyed_onto_an_admission_block(engine) -> None:
    other_operation_id = "00000000-0000-4000-8000-000000000012"
    with engine.begin() as connection:
        connection.execute(insert(guided_operation_admission_blocks_table).values(**_admission_block()))
        connection.execute(insert(guided_operations_table).values(**_operation(operation_id=other_operation_id)))
    with pytest.raises(IntegrityError, match="admission"), engine.begin() as connection:
        connection.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.operation_id == other_operation_id)
            .values(operation_id=OPERATION_ID)
        )


def test_admission_blocks_are_append_only_but_whole_session_cascade_is_allowed(engine) -> None:
    with engine.begin() as connection:
        connection.execute(insert(guided_operation_admission_blocks_table).values(**_admission_block()))

    with pytest.raises(IntegrityError, match="append-only"), engine.begin() as connection:
        connection.execute(update(guided_operation_admission_blocks_table).values(actor="tampered"))
    with pytest.raises(IntegrityError, match="append-only"), engine.begin() as connection:
        connection.execute(delete(guided_operation_admission_blocks_table))

    with engine.begin() as connection:
        connection.execute(delete(sessions_table).where(sessions_table.c.id == SESSION_ID))
        assert connection.execute(text("SELECT count(*) FROM guided_operation_admission_blocks")).scalar_one() == 0


def test_failed_event_accepts_exact_empty_commitment_storage_shape(engine) -> None:
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation()))
        connection.execute(
            insert(guided_operation_events_table).values(
                **_event(
                    event_kind="failed",
                    lease_expires_at=None,
                    failure_audit_cohort=_empty_failure_audit_cohort(),
                )
            )
        )


@pytest.mark.parametrize(
    "kind",
    [
        "guided_start",
        "guided_respond",
        "guided_chat",
        "guided_convert",
        "guided_reenter",
        "guided_plan",
        "state_revert",
        "session_fork",
    ],
)
def test_operation_kind_closed_vocabulary_accepts_supported_kinds(engine, kind: str) -> None:
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation(kind=kind, operation_id=f"{OPERATION_ID[:-1]}{kind[-1]}")))


@pytest.mark.parametrize(
    "column,value",
    [
        ("kind", "other"),
        ("status", "pending"),
        ("request_hash", "A" * 64),
        ("request_hash", "a" * 63),
        ("attempt", 0),
        ("operation_id", ""),
    ],
)
def test_operation_constraints_reject_invalid_values(engine, column: str, value: object) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation(**{column: value})))


def test_guided_plan_result_requires_exact_proposal_checkpoint_locator(engine) -> None:
    completed = _operation(
        kind="guided_plan",
        status="completed",
        lease_token=None,
        lease_expires_at=None,
        settled_at=NOW,
        result_kind="pipeline_proposal",
        proposal_id=PROPOSAL_ID,
        result_state_id=STATE_ID,
        response_hash=RESPONSE_HASH,
    )
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**completed))

    for missing in ("proposal_id", "result_state_id"):
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                insert(guided_operations_table).values(
                    **{**completed, "operation_id": f"{OPERATION_ID[:-1]}{1 if missing == 'proposal_id' else 2}", missing: None}
                )
            )


def test_request_cancelled_is_closed_terminal_failure(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    kind="guided_plan",
                    status="failed",
                    lease_token=None,
                    lease_expires_at=None,
                    settled_at=NOW,
                    failure_code="request_cancelled",
                )
            )
        )


@pytest.mark.parametrize(
    "values",
    [
        {"status": "in_progress", "lease_token": None},
        {"status": "in_progress", "lease_expires_at": None},
        {"status": "in_progress", "settled_at": NOW},
        {"status": "completed", "lease_token": "stale", "lease_expires_at": NOW},
        {
            "status": "completed",
            "settled_at": NOW,
            "result_kind": "composition_state",
            "result_state_id": None,
            "response_hash": RESPONSE_HASH,
        },
        {
            "status": "completed",
            "settled_at": NOW,
            "result_kind": "composition_state",
            "result_state_id": "00000000-0000-4000-8000-000000000004",
            "response_hash": None,
        },
        {"status": "failed", "settled_at": NOW, "failure_code": None, "lease_token": None, "lease_expires_at": None},
        {
            "status": "failed",
            "settled_at": NOW,
            "failure_code": "safe_failure",
            "response_hash": RESPONSE_HASH,
            "lease_token": None,
            "lease_expires_at": None,
        },
        {
            "status": "failed",
            "settled_at": NOW,
            "failure_code": "raw provider error: secret",
            "lease_token": None,
            "lease_expires_at": None,
        },
    ],
)
def test_status_bundles_are_exact(engine, values: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation(**values)))


def test_completed_and_failed_terminal_bundles_are_accepted(engine) -> None:
    completed_id = OPERATION_ID
    failed_id = "00000000-0000-4000-8000-000000000003"
    with engine.begin() as connection:
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id=completed_id,
                    status="completed",
                    lease_token=None,
                    lease_expires_at=None,
                    result_kind="composition_state",
                    result_state_id=STATE_ID,
                    response_hash=RESPONSE_HASH,
                    settled_at=NOW,
                )
            )
        )
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id=failed_id,
                    status="failed",
                    lease_token=None,
                    lease_expires_at=None,
                    failure_code="provider_unavailable",
                    settled_at=NOW,
                )
            )
        )
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id="00000000-0000-4000-8000-000000000005",
                    kind="session_fork",
                    status="completed",
                    lease_token=None,
                    lease_expires_at=None,
                    result_kind="session",
                    result_session_id=RESULT_SESSION_ID,
                    response_hash="c" * 64,
                    settled_at=NOW,
                )
            )
        )


def test_stale_conflict_is_a_current_closed_terminal_failure_code(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id="00000000-0000-4000-8000-000000000017",
                    status="failed",
                    lease_token=None,
                    lease_expires_at=None,
                    failure_code="stale_conflict",
                    settled_at=NOW,
                )
            )
        )


def test_in_progress_resumable_refs_are_kind_bound_but_supported(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id="00000000-0000-4000-8000-000000000011",
                    kind="session_fork",
                    result_session_id=RESULT_SESSION_ID,
                )
            )
        )
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id="00000000-0000-4000-8000-000000000014",
                    kind="guided_start",
                    result_state_id=STATE_ID,
                )
            )
        )


def test_in_progress_fork_cannot_bind_a_parent_session_state(engine) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id="00000000-0000-4000-8000-000000000016",
                    kind="session_fork",
                    result_state_id=STATE_ID,
                )
            )
        )


def test_operation_id_is_scoped_by_session(engine) -> None:
    second_session = "00000000-0000-4000-8000-000000000010"
    with engine.begin() as connection:
        connection.execute(
            insert(sessions_table).values(
                id=second_session,
                user_id="schema-test",
                auth_provider_type="local",
                title="Second",
                trust_mode="auto_commit",
                density_default="high",
                created_at=NOW,
                updated_at=NOW,
                interpretation_review_disabled=False,
            )
        )
        connection.execute(insert(guided_operations_table).values(**_operation()))
        connection.execute(insert(guided_operations_table).values(**_operation(session_id=second_session)))


@pytest.mark.parametrize(
    "values",
    [
        {"kind": "guided_start", "result_session_id": "00000000-0000-4000-8000-000000000009"},
        {"kind": "guided_start", "proposal_id": "00000000-0000-4000-8000-000000000008"},
        {"kind": "session_fork", "proposal_id": "00000000-0000-4000-8000-000000000008"},
        {"result_kind": "arbitrary_json"},
        {"result_kind": "composition_state", "result_state_id": "00000000-0000-4000-8000-000000000004"},
    ],
)
def test_in_progress_locator_fields_are_closed_and_kind_bound(engine, values: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation(**values)))


@pytest.mark.parametrize(
    "kind,result_kind,refs",
    [
        ("guided_start", "session", {"result_session_id": "00000000-0000-4000-8000-000000000009"}),
        ("guided_respond", "proposal", {"proposal_id": "00000000-0000-4000-8000-000000000008"}),
        ("guided_chat", "session", {"result_session_id": "00000000-0000-4000-8000-000000000009"}),
        ("guided_convert", "proposal", {"proposal_id": "00000000-0000-4000-8000-000000000008"}),
        ("guided_reenter", "session", {"result_session_id": "00000000-0000-4000-8000-000000000009"}),
        ("state_revert", "proposal", {"proposal_id": "00000000-0000-4000-8000-000000000008"}),
        ("session_fork", "composition_state", {"result_state_id": "00000000-0000-4000-8000-000000000004"}),
    ],
)
def test_completed_locator_discriminator_is_tied_to_operation_kind(
    engine,
    kind: str,
    result_kind: str,
    refs: dict[str, object],
) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    kind=kind,
                    status="completed",
                    lease_token=None,
                    lease_expires_at=None,
                    result_kind=result_kind,
                    response_hash=RESPONSE_HASH,
                    settled_at=NOW,
                    **refs,
                )
            )
        )


def test_terminal_operation_allows_settlement_once_then_is_immutable(engine) -> None:
    state_id = "00000000-0000-4000-8000-000000000004"
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation()))
        connection.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == SESSION_ID)
            .where(guided_operations_table.c.operation_id == OPERATION_ID)
            .values(lease_expires_at=NOW + timedelta(minutes=2), updated_at=NOW)
        )
        connection.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == SESSION_ID)
            .where(guided_operations_table.c.operation_id == OPERATION_ID)
            .values(
                status="completed",
                lease_token=None,
                lease_expires_at=None,
                result_kind="composition_state",
                result_state_id=state_id,
                response_hash=RESPONSE_HASH,
                settled_at=NOW,
            )
        )

    with pytest.raises(IntegrityError, match="terminal rows are immutable"), engine.begin() as connection:
        connection.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == SESSION_ID)
            .where(guided_operations_table.c.operation_id == OPERATION_ID)
            .values(response_hash="c" * 64)
        )


def test_failed_terminal_operation_is_immutable(engine) -> None:
    operation_id = "00000000-0000-4000-8000-000000000014"
    with engine.begin() as connection:
        connection.execute(
            insert(guided_operations_table).values(
                **_operation(
                    operation_id=operation_id,
                    status="failed",
                    lease_token=None,
                    lease_expires_at=None,
                    failure_code="operation_failed",
                    settled_at=NOW,
                )
            )
        )

    with pytest.raises(IntegrityError, match="terminal rows are immutable"), engine.begin() as connection:
        connection.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == SESSION_ID)
            .where(guided_operations_table.c.operation_id == operation_id)
            .values(failure_code="provider_timeout")
        )


def test_events_are_append_only_but_whole_session_cascade_is_allowed(engine) -> None:
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation()))
        connection.execute(insert(guided_operation_events_table).values(**_event()))

    with pytest.raises(IntegrityError, match="append-only"), engine.begin() as connection:
        connection.execute(update(guided_operation_events_table).values(actor="tampered"))
    with pytest.raises(IntegrityError, match="append-only"), engine.begin() as connection:
        connection.execute(delete(guided_operation_events_table))

    with engine.begin() as connection:
        connection.execute(delete(sessions_table).where(sessions_table.c.id == SESSION_ID))
        assert connection.execute(text("SELECT count(*) FROM guided_operation_events")).scalar_one() == 0


@pytest.mark.parametrize(
    "invalid",
    [
        _event(event_kind="other"),
        _event(sequence=0),
        _event(attempt=0),
        _event(event_kind="taken_over", attempt=2, prior_attempt=None),
        _event(event_kind="completed", lease_expires_at=NOW),
        _event(operation_id="00000000-0000-4000-8000-000000000099"),
        _event(request_hash="c" * 64),
    ],
    ids=["kind", "sequence", "attempt", "takeover-bundle", "terminal-bundle", "foreign-operation", "request-hash-binding"],
)
def test_event_constraints_and_same_operation_fk(engine, invalid: dict[str, object]) -> None:
    with engine.begin() as connection:
        connection.execute(insert(guided_operations_table).values(**_operation()))
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(guided_operation_events_table).values(**invalid))


@pytest.mark.parametrize(
    "trigger",
    [
        "trg_guided_operations_terminal_immutable",
        "trg_guided_operation_events_no_update",
        "trg_guided_operation_events_no_delete",
        "trg_guided_operation_admission_blocks_no_update",
        "trg_guided_operation_admission_blocks_no_delete",
        "trg_guided_operation_admission_blocks_reject_existing_operation",
        "trg_guided_operations_reject_admission_block_insert",
        "trg_guided_operations_reject_admission_block_update",
    ],
)
def test_startup_probe_rejects_missing_required_trigger(engine, trigger: str) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"DROP TRIGGER {trigger}"))
    with pytest.raises(SessionSchemaError, match="trigger"):
        initialize_session_schema(engine)


def test_schema_has_no_raw_request_or_lease_token_event_columns(engine) -> None:
    operation_columns = {column["name"] for column in inspect(engine).get_columns("guided_operations")}
    event_columns = {column["name"] for column in inspect(engine).get_columns("guided_operation_events")}
    block_columns = {column["name"] for column in inspect(engine).get_columns("guided_operation_admission_blocks")}
    assert "result_locator_json" not in operation_columns
    assert "result_kind" in operation_columns
    assert {"raw_body", "raw_intent", "provider_error"}.isdisjoint(operation_columns)
    assert {"lease_token", "raw_body", "raw_intent", "provider_error"}.isdisjoint(event_columns)
    assert {"request_hash", "lease_token", "raw_body", "raw_intent", "provider_error", "secret"}.isdisjoint(block_columns)
