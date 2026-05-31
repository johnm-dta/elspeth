"""GET /api/sessions/{sid}/messages include_tool_rows query parameter."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import insert

from elspeth.web.sessions.models import chat_messages_table
from tests.unit.web.conftest import _make_session


async def _get(test_client: TestClient, url: str) -> Response:
    async with AsyncClient(
        transport=ASGITransport(app=test_client.app),
        base_url="http://test",
        cookies=test_client.cookies,
    ) as client:
        response = await client.get(url)
        test_client.cookies.update(response.cookies)
        return response


def _seed_user_assistant_tool_rows(test_client: TestClient) -> dict[str, Any]:
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
                    "raw_content": None,
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


@pytest.mark.asyncio
async def test_default_excludes_tool_rows(test_client: TestClient) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)
    session_id = seeded["session_id"]

    response = await _get(test_client, f"/api/sessions/{session_id}/messages")

    assert response.status_code == 200
    rows = response.json()
    assert all(row["role"] in ("user", "assistant", "system") for row in rows)


@pytest.mark.asyncio
async def test_include_tool_rows_returns_tool_rows_interleaved_by_sequence_no(
    test_client: TestClient,
) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)
    session_id = seeded["session_id"]

    response = await _get(test_client, f"/api/sessions/{session_id}/messages?include_tool_rows=true")

    assert response.status_code == 200
    rows = response.json()
    sequence_nos = [row["sequence_no"] for row in rows]
    assert sequence_nos == sorted(sequence_nos)
    assert [row["role"] for row in rows] == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_include_tool_rows_exposes_tool_row_columns(
    test_client: TestClient,
) -> None:
    seeded = _seed_user_assistant_tool_rows(test_client)
    session_id = seeded["session_id"]

    response = await _get(test_client, f"/api/sessions/{session_id}/messages?include_tool_rows=true")

    assert response.status_code == 200
    rows = response.json()
    for row in rows:
        assert "sequence_no" in row
        if row["role"] == "tool":
            assert row["tool_call_id"] is not None
            assert row["parent_assistant_id"] == seeded["assistant_id"]
        else:
            assert row["tool_call_id"] is None
            assert row["parent_assistant_id"] is None
