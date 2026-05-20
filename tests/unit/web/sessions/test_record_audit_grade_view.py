"""record_audit_grade_view writes audit-grade transcript access rows."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import insert
from sqlalchemy.exc import SQLAlchemyError

from elspeth.web.sessions.models import chat_messages_table
from elspeth.web.sessions.protocol import AuditAccessLogWriteError
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value
from tests.unit.web.conftest import _make_session


async def _get(test_client: TestClient, url: str) -> Response:
    async with AsyncClient(
        transport=ASGITransport(
            app=test_client.app,
            raise_app_exceptions=getattr(test_client, "raise_server_exceptions", True),
        ),
        base_url="http://test",
        cookies=test_client.cookies,
    ) as client:
        response = await client.get(url)
        test_client.cookies.update(response.cookies)
        return response


@pytest.fixture
def sessions_service(engine, tmp_path: Path) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.record-audit-grade-view"),
    )


@pytest.fixture
def session_owned_by_alice(sessions_service: SessionServiceImpl) -> str:
    session_id = str(uuid4())
    with sessions_service._engine.begin() as conn:
        _make_session(conn, session_id=session_id, user_id="alice")
    return session_id


def _seed_user_assistant_tool_rows(test_client: TestClient, *, assistant_raw_content: str | None = None) -> dict[str, str]:
    session_id = str(uuid4())
    assistant_id = str(uuid4())
    now = datetime.now(UTC)
    with test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=session_id, user_id="alice")
        conn.execute(
            insert(chat_messages_table),
            [
                {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "role": "user",
                    "content": "Build it",
                    "raw_content": None,
                    "tool_calls": None,
                    "tool_call_id": None,
                    "sequence_no": 1,
                    "writer_principal": "route_user_message",
                    "created_at": now,
                    "composition_state_id": None,
                    "parent_assistant_id": None,
                },
                {
                    "id": assistant_id,
                    "session_id": session_id,
                    "role": "assistant",
                    "content": "Calling a tool",
                    "raw_content": assistant_raw_content,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_pipeline_state"}}],
                    "tool_call_id": None,
                    "sequence_no": 2,
                    "writer_principal": "compose_loop",
                    "created_at": now,
                    "composition_state_id": None,
                    "parent_assistant_id": None,
                },
                {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "role": "tool",
                    "content": "{}",
                    "raw_content": None,
                    "tool_calls": None,
                    "tool_call_id": "call_1",
                    "sequence_no": 3,
                    "writer_principal": "compose_loop",
                    "created_at": now,
                    "composition_state_id": None,
                    "parent_assistant_id": assistant_id,
                },
            ],
        )
    return {"session_id": session_id, "assistant_id": assistant_id}


def test_record_audit_grade_view_writes_row(
    sessions_service: SessionServiceImpl,
    session_owned_by_alice: str,
) -> None:
    sessions_service.record_audit_grade_view(
        session_id=session_owned_by_alice,
        requesting_principal="alice",
        request_path=f"/api/sessions/{session_owned_by_alice}/messages",
        query_args={"include_tool_rows": "true", "limit": "50"},
        ip_address="10.0.0.5",
    )

    rows = sessions_service.list_audit_access_log(session_id=session_owned_by_alice)

    assert len(rows) == 1
    row = rows[0]
    assert row.requesting_principal == "alice"
    assert row.writer_principal == "audit_grade_view"
    assert row.request_path == f"/api/sessions/{session_owned_by_alice}/messages"
    assert row.query_args == {"include_tool_rows": "true", "limit": "50"}
    assert row.ip_address == "10.0.0.5"


def test_record_audit_grade_view_increments_counter(
    sessions_service: SessionServiceImpl,
    session_owned_by_alice: str,
) -> None:
    sessions_service.record_audit_grade_view(
        session_id=session_owned_by_alice,
        requesting_principal="alice",
        request_path=f"/api/sessions/{session_owned_by_alice}/messages",
        query_args={},
        ip_address=None,
    )

    assert observed_value(sessions_service._telemetry.audit_grade_view_total) == 1


def test_record_audit_grade_view_writer_principal_is_pinned(sessions_service: SessionServiceImpl) -> None:
    sig = inspect.signature(sessions_service.record_audit_grade_view)

    assert "writer_principal" not in sig.parameters


def test_record_audit_grade_view_rejects_unallowlisted_query_args(
    sessions_service: SessionServiceImpl,
    session_owned_by_alice: str,
) -> None:
    with pytest.raises(ValueError, match="unallowlisted audit-grade query args"):
        sessions_service.record_audit_grade_view(
            session_id=session_owned_by_alice,
            requesting_principal="alice",
            request_path=f"/api/sessions/{session_owned_by_alice}/messages",
            query_args={"include_tool_rows": "true", "api_key": "secret"},
            ip_address=None,
        )


def test_record_audit_grade_view_translates_db_write_failure(
    sessions_service: SessionServiceImpl,
    session_owned_by_alice: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_begin():
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(sessions_service._engine, "begin", raise_begin)

    with pytest.raises(AuditAccessLogWriteError):
        sessions_service.record_audit_grade_view(
            session_id=session_owned_by_alice,
            requesting_principal="alice",
            request_path=f"/api/sessions/{session_owned_by_alice}/messages",
            query_args={"include_tool_rows": "true"},
            ip_address=None,
        )

    assert observed_value(sessions_service._telemetry.audit_access_log_write_failed_total) == 1
    assert observed_value(sessions_service._telemetry.audit_grade_view_total) == 0


@pytest.mark.asyncio
async def test_endpoint_emits_audit_log_when_include_tool_rows_true(test_client: TestClient) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)

    response = await _get(test_client, f"/api/sessions/{seeded['session_id']}/messages?include_tool_rows=true")

    assert response.status_code == 200
    sessions_service = test_client.app.state.session_service
    rows = sessions_service.list_audit_access_log(session_id=seeded["session_id"])
    assert len(rows) == 1
    assert rows[0].query_args == {"include_tool_rows": "true"}


@pytest.mark.asyncio
async def test_endpoint_emits_audit_log_when_include_llm_audit_true_without_tool_rows(test_client: TestClient) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)

    response = await _get(test_client, f"/api/sessions/{seeded['session_id']}/messages?include_llm_audit=true")

    assert response.status_code == 200
    sessions_service = test_client.app.state.session_service
    rows = sessions_service.list_audit_access_log(session_id=seeded["session_id"])
    assert len(rows) == 1
    assert rows[0].query_args == {"include_llm_audit": "true"}


@pytest.mark.asyncio
async def test_endpoint_emits_audit_log_when_include_raw_content_true_without_tool_rows(test_client: TestClient) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client, assistant_raw_content="provider final prose")

    response = await _get(test_client, f"/api/sessions/{seeded['session_id']}/messages?include_raw_content=true")

    assert response.status_code == 200
    body = response.json()
    assert [message["raw_content"] for message in body] == [None, "provider final prose"]
    sessions_service = test_client.app.state.session_service
    rows = sessions_service.list_audit_access_log(session_id=seeded["session_id"])
    assert len(rows) == 1
    assert rows[0].query_args == {"include_raw_content": "true"}


@pytest.mark.asyncio
async def test_endpoint_does_not_emit_audit_log_when_include_tool_rows_false(test_client: TestClient) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)

    response = await _get(test_client, f"/api/sessions/{seeded['session_id']}/messages")

    assert response.status_code == 200
    sessions_service = test_client.app.state.session_service
    rows = sessions_service.list_audit_access_log(session_id=seeded["session_id"])
    assert rows == []


@pytest.mark.asyncio
async def test_endpoint_filters_unallowlisted_query_args_before_audit_writer(
    test_client: TestClient,
) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)

    response = await _get(
        test_client,
        f"/api/sessions/{seeded['session_id']}/messages?include_tool_rows=true&api_key=secret&limit=25",
    )

    assert response.status_code == 200
    sessions_service = test_client.app.state.session_service
    rows = sessions_service.list_audit_access_log(session_id=seeded["session_id"])
    assert len(rows) == 1
    assert rows[0].query_args == {"include_tool_rows": "true", "limit": "25"}


@pytest.mark.asyncio
async def test_endpoint_records_combined_audit_grade_query_args(test_client: TestClient) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)

    response = await _get(
        test_client,
        (
            f"/api/sessions/{seeded['session_id']}/messages?"
            "include_tool_rows=true&include_llm_audit=true&include_raw_content=true&api_key=secret&limit=25"
        ),
    )

    assert response.status_code == 200
    sessions_service = test_client.app.state.session_service
    rows = sessions_service.list_audit_access_log(session_id=seeded["session_id"])
    assert len(rows) == 1
    assert rows[0].query_args == {
        "include_tool_rows": "true",
        "include_llm_audit": "true",
        "include_raw_content": "true",
        "limit": "25",
    }


@pytest.mark.asyncio
async def test_endpoint_fails_closed_when_audit_access_log_write_fails(
    test_client: TestClient,
    inject_audit_access_log_write_failure,
) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)
    sessions_service = test_client.app.state.session_service
    inject_audit_access_log_write_failure(sessions_service)
    test_client.raise_server_exceptions = False

    response = await _get(test_client, f"/api/sessions/{seeded['session_id']}/messages?include_tool_rows=true")

    assert response.status_code == 500
    body = response.json()
    assert body.get("error_type") == "audit_access_log_write_failed"
    assert "messages" not in body
    assert observed_value(sessions_service._telemetry.audit_access_log_write_failed_total) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("audit_query", ["include_llm_audit=true", "include_raw_content=true"])
async def test_endpoint_fails_closed_when_audit_access_log_write_fails_for_non_tool_audit_views(
    test_client: TestClient,
    inject_audit_access_log_write_failure,
    audit_query: str,
) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client, assistant_raw_content="provider final prose")
    sessions_service = test_client.app.state.session_service
    inject_audit_access_log_write_failure(sessions_service)
    test_client.raise_server_exceptions = False

    response = await _get(test_client, f"/api/sessions/{seeded['session_id']}/messages?{audit_query}")

    assert response.status_code == 500
    body = response.json()
    assert body.get("error_type") == "audit_access_log_write_failed"
    assert "messages" not in body
    assert observed_value(sessions_service._telemetry.audit_access_log_write_failed_total) == 1
