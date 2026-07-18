from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, insert, inspect, text, update
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    guided_operation_events_table,
    guided_operations_table,
    metadata,
    sessions_table,
)
from elspeth.web.sessions.schema import SessionSchemaError, initialize_session_schema

SESSION_ID = "00000000-0000-4000-8000-000000000001"
OPERATION_ID = "00000000-0000-4000-8000-000000000002"
REQUEST_HASH = "a" * 64
RESPONSE_HASH = "b" * 64
NOW = datetime(2026, 7, 18, tzinfo=UTC)


@pytest.fixture
def engine():
    engine = create_session_engine("sqlite:///:memory:")
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
        "result_state_id": None,
        "result_session_id": None,
        "result_locator_json": None,
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
        "occurred_at": NOW,
    }
    values.update(overrides)
    return values


def test_tables_and_composite_keys_are_current(engine) -> None:
    assert {"guided_operations", "guided_operation_events"} <= set(metadata.tables)
    inspector = inspect(engine)
    assert inspector.get_pk_constraint("guided_operations")["constrained_columns"] == ["session_id", "operation_id"]
    assert inspector.get_pk_constraint("guided_operation_events")["constrained_columns"] == [
        "session_id",
        "operation_id",
        "sequence",
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "guided_start",
        "guided_respond",
        "guided_chat",
        "guided_convert",
        "guided_reenter",
        "state_revert",
        "session_fork",
        "guided_plan",
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


@pytest.mark.parametrize(
    "values",
    [
        {"status": "in_progress", "lease_token": None},
        {"status": "in_progress", "lease_expires_at": None},
        {"status": "in_progress", "settled_at": NOW},
        {"status": "completed", "lease_token": "stale", "lease_expires_at": NOW},
        {"status": "completed", "settled_at": NOW, "result_locator_json": None, "response_hash": RESPONSE_HASH},
        {"status": "completed", "settled_at": NOW, "result_locator_json": "{}", "response_hash": None},
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
                    result_locator_json='{"state_id":"00000000-0000-4000-8000-000000000004"}',
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


def test_operation_id_is_scoped_by_session(engine) -> None:
    second_session = "00000000-0000-4000-8000-000000000009"
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


@pytest.mark.parametrize("trigger", ["trg_guided_operation_events_no_update", "trg_guided_operation_events_no_delete"])
def test_startup_probe_rejects_missing_event_trigger(engine, trigger: str) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"DROP TRIGGER {trigger}"))
    with pytest.raises(SessionSchemaError, match="trigger"):
        initialize_session_schema(engine)


def test_schema_has_no_raw_request_or_lease_token_event_columns(engine) -> None:
    operation_columns = {column["name"] for column in inspect(engine).get_columns("guided_operations")}
    event_columns = {column["name"] for column in inspect(engine).get_columns("guided_operation_events")}
    assert {"raw_body", "raw_intent", "provider_error"}.isdisjoint(operation_columns)
    assert {"lease_token", "raw_body", "raw_intent", "provider_error"}.isdisjoint(event_columns)
