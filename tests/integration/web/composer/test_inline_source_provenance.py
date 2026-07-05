"""Server-side creation_modality + LLM-provenance integration tests (Phase 5a.2.5).

These tests pin the audit-trail invariants the schema-and-writer change is
introducing on the ``blobs`` table.  They exercise the *real* call site
(:func:`elspeth.web.composer.tools._execute_set_pipeline` with a populated
``source.inline_blob``) so the assertions cover the entire wire path: the
LLM-supplied inline blob → :func:`_prepare_blob_create` → row insert →
post-insert read.  Test #5 (non-UTF-8 surrogate) targets
:func:`_prepare_blob_create` directly per the spec's
"surrogate test bypasses the route layer" exception.

Test list (1-5):
  1. ``test_verbatim_blob_records_creation_modality_and_message_id``
  2. ``test_llm_generated_blob_carries_llm_provenance``
  3. ``test_llm_generated_blob_requires_message_anchor``
  4. ``test_cross_session_message_id_rejected``
  5. ``test_oversize_content_raises_tool_argument_error``
  6. ``test_non_utf8_content_raises_tool_argument_error``

Layer: integration — boots an in-memory session DB via
``create_session_engine`` + ``initialize_session_schema``, then drives
``execute_tool('set_pipeline', ...)`` against a real ``BlobsTable``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.catalog.schemas import PluginSchemaInfo
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _prepare_blob_create, execute_tool
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    blobs_table,
    chat_messages_table,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema

# ─────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────

_USER_MESSAGE_CONTENT = "Use this CSV: name,score\nada,42\n"


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


class _CatalogFake:
    def has_plugin(self, _plugin_type: str, _name: str) -> bool:
        return True

    def get_schema(self, _plugin_type: str, _name: str) -> PluginSchemaInfo:
        return PluginSchemaInfo(
            name="csv",
            plugin_type="source",
            description="CSV file source",
            json_schema={"title": "CsvSourceConfig", "properties": {"path": {"type": "string"}}},
            knob_schema={"fields": []},
        )


def _mock_catalog() -> _CatalogFake:
    return _CatalogFake()


def _session_with_user_message() -> tuple[Any, str, str]:
    """Boot a session DB and seed a session + one user chat message.

    Returns (engine, session_id, user_message_id).  The user message
    plays the role of the "triggering message" the inline blob's
    provenance binds to.
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
                title="Phase 5a.2.5 test session",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content=_USER_MESSAGE_CONTENT,
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
    return engine, session_id, user_message_id


def _minimal_inline_blob_args(content: str) -> dict[str, Any]:
    """Build the smallest ``set_pipeline`` args dict that carries an inline blob.

    The ``schema: {mode: observed}`` option is required by the CSV source's
    pre-validation (the plugin requires explicit schema declaration). We
    use the observed-mode default since the inline blob's columns are
    not known ahead of time at the composer layer.
    """
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"schema": {"mode": "observed"}},
            "inline_blob": {
                "filename": "ada.csv",
                "mime_type": "text/csv",
                "content": content,
            },
        },
        "nodes": [],
        "edges": [],
        "outputs": [],
    }


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — verbatim path records modality + message id
# ─────────────────────────────────────────────────────────────────────────


def test_verbatim_blob_records_creation_modality_and_message_id(tmp_path: Path) -> None:
    """A set_pipeline.inline_blob produced from a verbatim user message must
    record ``creation_modality='verbatim'``, the originating ``user_message_id``,
    and NULL across all five ``creating_*`` LLM-provenance fields.
    """
    engine, session_id, user_message_id = _session_with_user_message()
    args = _minimal_inline_blob_args("name,score\nada,42\n")

    result = execute_tool(
        "set_pipeline",
        args,
        _empty_state(),
        _mock_catalog(),
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        user_message_id=user_message_id,
        user_message_content=_USER_MESSAGE_CONTENT,
    )
    assert result.success is True, result.data

    with engine.begin() as conn:
        row = conn.execute(select(blobs_table).where(blobs_table.c.session_id == session_id)).one()

    assert row.creation_modality == CreationModality.VERBATIM.value
    assert row.created_from_message_id == user_message_id
    assert row.creating_model_identifier is None
    assert row.creating_model_version is None
    assert row.creating_provider is None
    assert row.creating_composer_skill_hash is None
    assert row.creating_arguments_hash is None


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — LLM-provenance round-trips through _prepare_blob_create
# ─────────────────────────────────────────────────────────────────────────


def test_llm_generated_blob_carries_llm_provenance(tmp_path: Path) -> None:
    """When a caller marks an inline blob ``creation_modality=LLM_GENERATED``
    and supplies the five ``creating_*`` fields, the prepared payload must
    carry every field through to the DB row.

    Drives :func:`_prepare_blob_create` directly: the call-site classifier
    that distinguishes verbatim-from-user-message vs LLM-authored content
    is deferred to a later Phase 5a task (3/4/8 per the plan), so the
    integration surface today is the writer contract on the helper,
    not a discriminant in ``_execute_set_pipeline``.  Persisting via the
    same engine + insert path the production handler uses keeps this an
    integration test, not a unit test.
    """
    from elspeth.web.composer.tools import _persist_prepared_blob_create

    engine, session_id, user_message_id = _session_with_user_message()

    prepared = _prepare_blob_create(
        {
            "filename": "generated.csv",
            "mime_type": "text/csv",
            "content": "score\n42\n",
        },
        data_dir=str(tmp_path),
        session_id=session_id,
        creation_modality=CreationModality.LLM_GENERATED,
        created_from_message_id=user_message_id,
        creating_model_identifier="gpt-5.4-mini",
        creating_model_version="2026-05-01",
        creating_provider="openai",
        creating_composer_skill_hash="sha256:cafebabe" + "0" * 56,
        creating_arguments_hash="sha256:deadbeef" + "0" * 56,
    )
    quota_error = _persist_prepared_blob_create(
        prepared,
        session_engine=engine,
        session_id=session_id,
    )
    assert quota_error is None

    with engine.begin() as conn:
        row = conn.execute(select(blobs_table).where(blobs_table.c.id == prepared.blob_id)).one()

    assert row.creation_modality == CreationModality.LLM_GENERATED.value
    assert row.created_from_message_id == user_message_id
    assert row.creating_model_identifier == "gpt-5.4-mini"
    assert row.creating_model_version == "2026-05-01"
    assert row.creating_provider == "openai"
    assert row.creating_composer_skill_hash.startswith("sha256:cafebabe")
    assert row.creating_arguments_hash.startswith("sha256:deadbeef")


def test_llm_generated_blob_requires_message_anchor(tmp_path: Path) -> None:
    """LLM-authored blob writes must name the user message that triggered them."""
    _engine, session_id, _user_message_id = _session_with_user_message()

    with pytest.raises(AuditIntegrityError, match="created_from_message_id"):
        _prepare_blob_create(
            {
                "filename": "generated.csv",
                "mime_type": "text/csv",
                "content": "score\n42\n",
            },
            data_dir=str(tmp_path),
            session_id=session_id,
            creation_modality=CreationModality.LLM_GENERATED,
            created_from_message_id=None,
            creating_model_identifier="gpt-5.4-mini",
            creating_model_version="2026-05-01",
            creating_provider="openai",
            creating_composer_skill_hash="sha256:cafebabe" + "0" * 56,
            creating_arguments_hash="sha256:deadbeef" + "0" * 56,
        )


# ─────────────────────────────────────────────────────────────────────────
# Test 4 — cross-session message id is rejected by the composite FK
# ─────────────────────────────────────────────────────────────────────────


def test_cross_session_message_id_rejected(tmp_path: Path) -> None:
    """The composite ``(created_from_message_id, session_id)`` FK must
    reject a blob whose ``created_from_message_id`` references a message
    in a different session.  This is the audit-trail anti-IDOR closure:
    the schema mechanically prevents the recorder from binding a blob to
    a message it does not actually descend from.
    """
    from elspeth.web.composer.tools import _persist_prepared_blob_create

    # Session A with its own user message.
    engine, session_a_id, _ = _session_with_user_message()

    # Add a SECOND session (B) on the SAME engine with its own user message.
    session_b_id = str(uuid4())
    session_b_message_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_b_id,
                user_id="bob",
                auth_provider_type="local",
                title="Other session",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=session_b_message_id,
                session_id=session_b_id,
                role="user",
                content="Different session message",
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

    # Try to persist a blob in session A whose created_from_message_id
    # references session B's message.  The composite FK must reject this.
    prepared = _prepare_blob_create(
        {
            "filename": "cross.csv",
            "mime_type": "text/csv",
            "content": "x\n",
        },
        data_dir=str(tmp_path),
        session_id=session_a_id,
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id=session_b_message_id,
    )
    with pytest.raises(IntegrityError):
        _persist_prepared_blob_create(
            prepared,
            session_engine=engine,
            session_id=session_a_id,
        )


# ─────────────────────────────────────────────────────────────────────────
# Test 4 — content larger than the cap is rejected at the Pydantic boundary
# ─────────────────────────────────────────────────────────────────────────


def test_oversize_content_raises_tool_argument_error(tmp_path: Path) -> None:
    """A 300 KiB inline blob content exceeds the ``_InlineBlobModel.content``
    256 KiB max_length cap.  The error must surface as ``ToolArgumentError``
    at the compose-loop boundary (CEC1 channel discipline) — not as a bare
    ``PydanticValidationError`` escape.
    """
    engine, session_id, user_message_id = _session_with_user_message()
    oversize_content = "x" * (300 * 1024)  # 300 KiB > 256 KiB cap
    args = _minimal_inline_blob_args(oversize_content)

    with pytest.raises(ToolArgumentError):
        execute_tool(
            "set_pipeline",
            args,
            _empty_state(),
            _mock_catalog(),
            data_dir=str(tmp_path),
            session_engine=engine,
            session_id=session_id,
            user_message_id=user_message_id,
        )


# ─────────────────────────────────────────────────────────────────────────
# Test 5 — non-UTF-8 (surrogate) content is rejected
# ─────────────────────────────────────────────────────────────────────────


def test_non_utf8_content_raises_tool_argument_error(tmp_path: Path) -> None:
    """A string containing an unpaired surrogate cannot be UTF-8 encoded;
    :func:`_prepare_blob_create` must raise ``ToolArgumentError`` rather
    than letting the bare ``UnicodeEncodeError`` escape into the audit
    layer.  The error message must mention ``valid UTF-8 text``.
    """
    session_id = str(uuid4())
    surrogate_content = "ok\udc80broken"  # lone low-surrogate code point

    with pytest.raises(ToolArgumentError) as exc_info:
        _prepare_blob_create(
            {
                "filename": "bad.txt",
                "mime_type": "text/plain",
                "content": surrogate_content,
            },
            data_dir=str(tmp_path),
            session_id=session_id,
            creation_modality=CreationModality.VERBATIM,
            created_from_message_id=None,
        )
    assert "valid UTF-8 text" in str(exc_info.value)
