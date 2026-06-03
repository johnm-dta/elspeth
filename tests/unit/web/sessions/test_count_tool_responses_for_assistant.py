"""count_tool_responses_for_assistant read helper (spec §6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from sqlalchemy import insert

from elspeth.web.sessions.models import chat_messages_table
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web.conftest import _make_session


@dataclass(frozen=True, slots=True)
class _PersistedAssistant:
    session_id: str
    id: str


@pytest.fixture
def sessions_service(engine, tmp_path: Path) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.count-tool-responses"),
    )


def _persist_assistant_with_tools(
    service: SessionServiceImpl,
    *,
    tool_count: int,
) -> _PersistedAssistant:
    session_id = str(uuid4())
    assistant_id = str(uuid4())
    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=session_id)
        conn.execute(
            insert(chat_messages_table).values(
                id=assistant_id,
                session_id=session_id,
                role="assistant",
                content="assistant",
                raw_content=None,
                tool_calls=[{"id": f"call_{idx}", "type": "function"} for idx in range(tool_count)] or None,
                sequence_no=1,
                writer_principal="compose_loop",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=now,
            )
        )
        for idx in range(tool_count):
            conn.execute(
                insert(chat_messages_table).values(
                    id=str(uuid4()),
                    session_id=session_id,
                    role="tool",
                    content="{}",
                    raw_content=None,
                    tool_calls=None,
                    sequence_no=idx + 2,
                    writer_principal="compose_loop",
                    composition_state_id=None,
                    tool_call_id=f"call_{idx}",
                    parent_assistant_id=assistant_id,
                    created_at=now,
                )
            )
    return _PersistedAssistant(session_id=session_id, id=assistant_id)


def test_count_zero_when_no_tool_rows(sessions_service: SessionServiceImpl) -> None:
    assistant = _persist_assistant_with_tools(sessions_service, tool_count=0)

    count = sessions_service.count_tool_responses_for_assistant(
        session_id=assistant.session_id,
        assistant_message_id=assistant.id,
    )

    assert count == 0


def test_count_matches_inserted_tool_rows(sessions_service: SessionServiceImpl) -> None:
    assistant = _persist_assistant_with_tools(sessions_service, tool_count=3)

    count = sessions_service.count_tool_responses_for_assistant(
        session_id=assistant.session_id,
        assistant_message_id=assistant.id,
    )

    assert count == 3


def test_count_none_assistant_id_returns_zero(sessions_service: SessionServiceImpl) -> None:
    count = sessions_service.count_tool_responses_for_assistant(
        session_id="session_1",
        assistant_message_id=None,
    )

    assert count == 0


@pytest.mark.asyncio
async def test_async_dispatcher_runs_in_worker_thread(sessions_service: SessionServiceImpl) -> None:
    assistant = _persist_assistant_with_tools(sessions_service, tool_count=3)

    count = await sessions_service.count_tool_responses_for_assistant_async(
        session_id=assistant.session_id,
        assistant_message_id=assistant.id,
    )

    assert count == 3
