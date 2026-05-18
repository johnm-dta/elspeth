"""Phase 5a attributability chain is tamper-evident (Task 2.6).

Walks ``blob.created_from_message_id`` → ``chat_messages.id`` and asserts the
two anchoring invariants the audit trail depends on:

- **2b** ``test_blob_provenance_anchor_is_immutable`` — mutation attempts on
  ``chat_messages.content`` raise :class:`IntegrityError` via the
  ``trg_chat_messages_immutable_content`` trigger owned by
  ``18a-phase-5b-backend.md`` (F-4).

- **2c** ``test_chat_message_delete_while_blob_references_it_raises`` —
  attempting to DELETE the ``chat_messages`` row a blob points at raises
  :class:`IntegrityError` via the unconditional
  ``trg_chat_messages_no_delete`` trigger.

The fixture pattern mirrors
``test_inline_source_provenance.py``: a function-local helper boots an
in-memory session DB via :func:`create_session_engine`, seeds one
``sessions`` row and one ``chat_messages`` row, and returns the engine plus
the IDs.  The blob row is then created through the *real* call site
(:func:`elspeth.web.composer.tools._prepare_blob_create` +
:func:`_persist_prepared_blob_create`) so the FK relationship under test is
the one production code writes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.enums import CreationModality
from elspeth.web.composer.tools import _persist_prepared_blob_create, _prepare_blob_create
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


def _session_with_user_message_and_blob(tmp_path: Path) -> tuple[Any, str, str]:
    """Boot a session DB, seed one session + one user chat message, and
    persist a blob whose ``created_from_message_id`` binds to that message.

    Returns ``(engine, session_id, user_message_id)``.  The blob row is
    written via the real composer write path so the composite FK
    ``fk_blobs_created_from_message_session`` is exercised end-to-end.
    """
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    user_message_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="alice",
                auth_provider_type="local",
                title="Phase 5a.2.6 attributability test session",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content="Use this CSV: name,score\\nada,42\\n",
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )

    prepared = _prepare_blob_create(
        {
            "filename": "ada.csv",
            "mime_type": "text/csv",
            "content": "name,score\nada,42\n",
        },
        data_dir=str(tmp_path),
        session_id=session_id,
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id=user_message_id,
    )
    quota_error = _persist_prepared_blob_create(
        prepared,
        session_engine=engine,
        session_id=session_id,
    )
    assert quota_error is None

    return engine, session_id, user_message_id


def test_blob_provenance_anchor_is_immutable(tmp_path: Path) -> None:
    """``blob.created_from_message_id`` points to an immutable ``chat_messages`` row.

    The ``trg_chat_messages_immutable_content`` trigger is part of current
    schema bootstrap; if it is missing, this test must fail rather than skip.
    """
    engine, session_id, user_message_id = _session_with_user_message_and_blob(tmp_path)

    # Confirm the blob's created_from_message_id resolves to the expected row.
    with engine.connect() as conn:
        blob_row = conn.execute(
            text("SELECT created_from_message_id FROM blobs WHERE session_id = :sid LIMIT 1"),
            {"sid": session_id},
        ).fetchone()
    assert blob_row is not None
    assert blob_row[0] == user_message_id

    # Mutation of ``content`` must raise IntegrityError — trigger fires.
    with pytest.raises(IntegrityError, match="append-only"), engine.begin() as conn:
        conn.execute(
            text("UPDATE chat_messages SET content = 'tampered' WHERE id = :id"),
            {"id": user_message_id},
        )


def test_chat_message_delete_while_blob_references_it_raises(tmp_path: Path) -> None:
    """DELETE on a ``chat_messages`` row referenced by a blob raises ``IntegrityError``.

    Protection comes from ``trg_chat_messages_no_delete``. The FK also
    protects referenced rows, but the trigger is deliberately stronger:
    every chat message is an audit anchor, even before a blob points at it.
    """
    engine, _session_id, user_message_id = _session_with_user_message_and_blob(tmp_path)

    with (
        pytest.raises(IntegrityError, match="append-only"),
        engine.begin() as conn,
    ):
        conn.execute(
            text("DELETE FROM chat_messages WHERE id = :id"),
            {"id": user_message_id},
        )
