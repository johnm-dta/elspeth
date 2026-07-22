"""ARG_ERROR routing + redaction for ``update_blob`` (Task 13 / Wave 2).

Companion to ``test_promote_create_blob.py``; same Wave 2 discipline
(rev-2 BLOCKER_A + M.10).  ``update_blob`` shares the
``_summarize_inline_blob_content`` summarizer with ``create_blob`` —
both blob-mutation tools accept the same payload shape and the
audit-side redaction MUST be uniform across them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, insert, select
from sqlalchemy.pool import StaticPool

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    MANIFEST,
    UpdateBlobArgumentsModel,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _execute_create_blob, _execute_update_blob
from elspeth.web.composer.tools._common import ToolContext as _ToolContext
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _mock_catalog() -> MagicMock:
    return MagicMock(spec=CatalogService)


def ToolContext(*, catalog: CatalogService, **kwargs: Any) -> _ToolContext:
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return _ToolContext(
        catalog=PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
        **kwargs,
    )


def _session_engine_with_session() -> tuple[Any, str]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                title="Test Session",
                created_at=now,
                updated_at=now,
            )
        )
    return engine, session_id


def _insert_user_message(engine: Any, session_id: str, content: str) -> str:
    user_message_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        sequence_no = (
            int(
                conn.execute(
                    select(func.coalesce(func.max(chat_messages_table.c.sequence_no), 0)).where(
                        chat_messages_table.c.session_id == session_id
                    )
                ).scalar_one()
            )
            + 1
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content=content,
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=sequence_no,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return user_message_id


# ---------------------------------------------------------------------------
# Manifest shape pin
# ---------------------------------------------------------------------------


def test_update_blob_manifest_entry_is_type_driven() -> None:
    entry = MANIFEST["update_blob"]
    assert entry.argument_model is UpdateBlobArgumentsModel
    assert entry.policy is None


# ---------------------------------------------------------------------------
# ARG_ERROR routing — bare ValidationError must NOT escape the handler
# ---------------------------------------------------------------------------


class TestPromoteUpdateBlobArgErrorRouting:
    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """A bare ``{}`` is missing both required fields."""
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_update_blob(
                {},
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_wrong_type_content_raises_tool_argument_error(self) -> None:
        """Pydantic rejects ``content: int`` before the handler acquires the session lock.

        Validation MUST run before the file-mutation critical section
        (per the handler docstring): if it didn't, the rollback path
        would issue an unnecessary filesystem write over an unmodified
        file on a pure argument-validation failure.
        """
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_update_blob(
                {"blob_id": "anything", "content": 42},
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_blob_id_raises_tool_argument_error(self) -> None:
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_update_blob(
                {"content": "new content"},
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' rejects fields belonging to neighbouring tools."""
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_update_blob(
                {
                    "blob_id": "anything",
                    "content": "hello",
                    "filename": "x.txt",  # belongs on create_blob
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_valid_arguments_dispatch_normally(self, tmp_path: Path) -> None:
        """Functional smoke: a valid call updates an existing blob's content."""
        engine, session_id = _session_engine_with_session()
        catalog = _mock_catalog()

        # Bootstrap an existing blob via create_blob (also a Task 13 promoted
        # tool — exercises the create_blob → update_blob lifecycle together).
        user_message_content = "Use this exact content:\nold"
        user_message_id = _insert_user_message(engine, session_id, user_message_content)
        create_result = _execute_create_blob(
            {"filename": "seed.txt", "mime_type": "text/plain", "content": "old"},
            _empty_state(),
            ToolContext(
                catalog=catalog,
                data_dir=str(tmp_path),
                session_engine=engine,
                session_id=session_id,
                user_message_id=user_message_id,
                user_message_content=user_message_content,
            ),
        )
        assert create_result.success is True
        blob_id = create_result.data["blob_id"]

        update_user_message_content = "Use this exact content:\nnew contents"
        update_user_message_id = _insert_user_message(engine, session_id, update_user_message_content)
        update_result = _execute_update_blob(
            {"blob_id": blob_id, "content": "new contents"},
            _empty_state(),
            ToolContext(
                catalog=catalog,
                data_dir=str(tmp_path),
                session_engine=engine,
                session_id=session_id,
                user_message_id=update_user_message_id,
                user_message_content=update_user_message_content,
            ),
        )
        assert update_result.success is True
        assert update_result.data is not None
        # update_blob's data payload carries the updated size_bytes /
        # content_hash; we verify byte count matches the new content.
        assert update_result.data["size_bytes"] == len(b"new contents")


# ---------------------------------------------------------------------------
# Redaction at the persistence boundary
# ---------------------------------------------------------------------------


_CANARY = "CANARY-UPDATE-BLOB-CONTENT-DO-NOT-LEAK"


def test_redaction_substitutes_content_via_summarizer() -> None:
    """``content`` is replaced by the ``<inline-blob:N-bytes>`` summarizer.

    Mirrors :func:`test_redaction_substitutes_content_via_summarizer`
    in ``test_promote_create_blob.py`` — the redaction surface for the
    two blob-mutation tools is intentionally uniform (both consume the
    same payload shape; the audit-side redaction must not diverge).
    """
    tel = NoopRedactionTelemetry()
    args = {
        "blob_id": "some-id",
        "content": _CANARY,
    }
    redacted = redact_tool_call_arguments("update_blob", args, telemetry=tel)
    # Structural keys preserved (blob_id is a reference, not sensitive).
    assert redacted["blob_id"] == "some-id"
    # Sensitive substitution.
    expected = f"<inline-blob:{len(_CANARY.encode('utf-8'))}-bytes>"
    assert redacted["content"] == expected
    # Canary MUST NOT appear anywhere in the redacted dict or its JSON form.
    serialized = json.dumps(redacted, sort_keys=True)
    assert _CANARY not in serialized
    # Telemetry recorded the manifest dispatch with the type-driven shape.
    assert tel.manifest_dispatch_calls == [{"tool_name": "update_blob", "shape": "type_driven"}]


# ---------------------------------------------------------------------------
# Blob-tool summarizer type-variability (rev-2 M.10)
# ---------------------------------------------------------------------------


class TestSummarizerTypeVariability:
    """Pin rev-2 M.10 for ``update_blob``.

    The summarizer is shared with ``create_blob``
    (:func:`_summarize_inline_blob_content`) and exhaustively tested
    in ``test_promote_create_blob.py``.  Here we pin the
    model-validation side of the contract: Pydantic's
    ``content: str`` + ``extra="forbid"`` rejects non-``str`` values
    BEFORE the summarizer is reached, so the lambda's ``str``-only
    safety is mechanically enforced for ``update_blob`` too.

    Duplicating the rejection checks (rather than referencing the
    create_blob test) is deliberate: the two models are independent
    raise sites and one should be moveable without the other (same
    discipline as :class:`TestUpdateBlobTypeGuard` / :class:`TestCreateBlobTypeGuard`
    in ``test_tools.py``).
    """

    def test_pydantic_model_rejects_none_content(self) -> None:
        with pytest.raises(PydanticValidationError):
            UpdateBlobArgumentsModel.model_validate({"blob_id": "anything", "content": None})

    def test_pydantic_model_rejects_non_string_content(self) -> None:
        with pytest.raises(PydanticValidationError):
            UpdateBlobArgumentsModel.model_validate({"blob_id": "anything", "content": 42})

    def test_pydantic_model_accepts_empty_string(self) -> None:
        """An empty-string ``content`` is a valid (if unusual) Tier-3 input.

        The summarizer formats it as ``<inline-blob:0-bytes>`` —
        exhaustively tested in ``test_promote_create_blob.py``
        :class:`TestSummarizerTypeVariability`.
        """
        validated = UpdateBlobArgumentsModel.model_validate({"blob_id": "anything", "content": ""})
        assert validated.content == ""

    def test_pydantic_model_accepts_non_ascii_string(self) -> None:
        validated = UpdateBlobArgumentsModel.model_validate({"blob_id": "anything", "content": "héllo"})
        assert validated.content == "héllo"
