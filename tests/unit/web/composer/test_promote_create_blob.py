"""ARG_ERROR routing + redaction for ``create_blob`` (Task 13 / Wave 2).

Companion to ``test_promote_set_source.py`` (Task 4); rev-2 BLOCKER_A
applies: promoted handlers MUST catch :class:`pydantic.ValidationError`
and re-raise as :class:`ToolArgumentError`.  A bare ``ValidationError``
escaping the handler hits ``service.py:2564`` (→
:class:`ComposerPluginCrashError` → HTTP 500) — wrong disposition for
Tier-3 input.

Tests pin:
  * exception-class + ``__cause__`` chain on invalid arguments,
  * valid dispatch produces a ready blob (functional smoke),
  * :func:`redact_tool_call_arguments` substitutes the
    ``<inline-blob:N-bytes>`` summarizer at the persistence boundary,
  * blob-tool summarizer type-variability (rev-2 M.10): the summarizer
    is safe for every ``str`` value Pydantic admits (empty, ASCII,
    non-ASCII) because ``extra="forbid"`` + ``content: str`` rejects
    everything but ``str`` BEFORE the summarizer runs.
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
from sqlalchemy import insert, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    MANIFEST,
    CreateBlobArgumentsModel,
    _summarize_inline_blob_content,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _execute_create_blob
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.composer.tools.blobs import _blob_creation_provenance
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, chat_messages_table, sessions_table
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
    # _execute_create_blob does not read from the catalog; a spec'd MagicMock
    # is sufficient.  This mirrors test_promote_set_source.py's discipline:
    # don't instantiate the real CatalogService here unless the path under
    # test actually consults it.  spec=CatalogService keeps the mock honest
    # against future protocol changes (I8 — test-analyst review remediation).
    return MagicMock(spec=CatalogService)


def _session_engine_with_session() -> tuple[Any, str]:
    """Minimal session DB with one row, matching test_tools.py helper."""
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


def _session_engine_with_user_message(content: str) -> tuple[Any, str, str]:
    """Minimal session DB seeded with the triggering user message."""
    engine, session_id = _session_engine_with_session()
    user_message_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(chat_messages_table).values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content=content,
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


# ---------------------------------------------------------------------------
# Manifest shape pin (parity with test_redact_set_source.py)
# ---------------------------------------------------------------------------


def test_create_blob_manifest_entry_is_type_driven() -> None:
    entry = MANIFEST["create_blob"]
    assert entry.argument_model is CreateBlobArgumentsModel
    assert entry.policy is None


# ---------------------------------------------------------------------------
# ARG_ERROR routing — bare ValidationError must NOT escape the handler
# ---------------------------------------------------------------------------


class TestPromoteCreateBlobArgErrorRouting:
    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """A bare ``{}`` is missing all three required fields."""
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_create_blob(
                {},
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    data_dir="/tmp",
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        # __cause__ chain MUST preserve the underlying ValidationError
        # so auditors can inspect missing fields without the LLM-facing
        # message exposing raw Tier-3 argument values.
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_wrong_type_raises_tool_argument_error(self) -> None:
        """Pydantic rejects ``content: int`` before the handler reads it."""
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_create_blob(
                {
                    "filename": "seed.txt",
                    "mime_type": "text/plain",
                    "content": 42,  # not str
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    data_dir="/tmp",
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' on the model rejects stray fields at Tier-3."""
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_create_blob(
                {
                    "filename": "seed.txt",
                    "mime_type": "text/plain",
                    "content": "hello",
                    "blob_id": "stray",  # belongs on update_blob / set_source_from_blob
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    data_dir="/tmp",
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_valid_arguments_dispatch_normally(self, tmp_path: Path) -> None:
        """Functional smoke: a valid call produces a ready blob."""
        user_message_content = "Please use this exact content:\nhello world"
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        result = _execute_create_blob(
            {
                "filename": "seed.txt",
                "mime_type": "text/plain",
                "content": "hello world",
            },
            _empty_state(),
            ToolContext(
                catalog=_mock_catalog(),
                data_dir=str(tmp_path),
                session_engine=engine,
                session_id=session_id,
                user_message_id=user_message_id,
                user_message_content=user_message_content,
            ),
        )
        assert result.success is True
        assert result.data is not None
        assert result.data["filename"] == "seed.txt"
        assert result.data["mime_type"] == "text/plain"
        assert result.data["size_bytes"] == len(b"hello world")


class TestCreateBlobComposerSourceProvenance:
    def test_no_message_and_no_composer_provenance_fails_closed(self, tmp_path: Path) -> None:
        """Blob writes without a message anchor or composer provenance must not persist as verbatim."""
        engine, session_id = _session_engine_with_session()

        with pytest.raises(AuditIntegrityError, match="missing: user_message_id"):
            _execute_create_blob(
                {
                    "filename": "generated.csv",
                    "mime_type": "text/csv",
                    "content": "priority,score\nhigh,99\n",
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    data_dir=str(tmp_path),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )

        with engine.connect() as conn:
            rows = conn.execute(select(blobs_table).where(blobs_table.c.session_id == session_id)).fetchall()
        assert rows == []

    @pytest.mark.parametrize("missing_user_message_id", [None, "", "   "])
    def test_contained_content_without_message_id_fails_closed(
        self,
        tmp_path: Path,
        missing_user_message_id: str | None,
    ) -> None:
        """Verbatim containment still requires a persisted message anchor."""
        engine, session_id = _session_engine_with_session()
        user_message_content = "Use this exact CSV:\npriority,score\nhigh,99\n"

        with pytest.raises(AuditIntegrityError, match="missing: user_message_id"):
            _execute_create_blob(
                {
                    "filename": "verbatim.csv",
                    "mime_type": "text/csv",
                    "content": "priority,score\nhigh,99\n",
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    data_dir=str(tmp_path),
                    session_engine=engine,
                    session_id=session_id,
                    user_message_id=missing_user_message_id,
                    user_message_content=user_message_content,
                ),
            )

        with engine.connect() as conn:
            rows = conn.execute(select(blobs_table).where(blobs_table.c.session_id == session_id)).fetchall()
        assert rows == []

    def test_empty_content_without_composer_provenance_fails_closed(self) -> None:
        """Empty content is valid locally but can never prove verbatim containment."""
        with pytest.raises(AuditIntegrityError, match="missing: user_message_id"):
            _blob_creation_provenance(
                "",
                ToolContext(
                    catalog=_mock_catalog(),
                    user_message_content="User supplied an empty file.",
                ),
            )

    def test_content_not_in_user_message_records_llm_generated_provenance(self, tmp_path: Path) -> None:
        """LLM-authored blob content must mechanically carry composer provenance."""
        engine, session_id, user_message_id = _session_engine_with_user_message("Please create a CSV of the high-priority records.")

        result = _execute_create_blob(
            {
                "filename": "generated.csv",
                "mime_type": "text/csv",
                "content": "priority,score\nhigh,99\n",
            },
            _empty_state(),
            ToolContext(
                catalog=_mock_catalog(),
                data_dir=str(tmp_path),
                session_engine=engine,
                session_id=session_id,
                user_message_id=user_message_id,
                user_message_content="Please create a CSV of the high-priority records.",
                composer_model_identifier="openai/gpt-5-mini",
                composer_model_version="gpt-5-mini-2026-05-01",
                composer_provider="openai",
                composer_skill_hash="sha256:composer-skill",
                tool_arguments_hash="sha256:tool-arguments",
            ),
        )

        assert result.success is True
        with engine.connect() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == result.data["blob_id"])).one()

        assert row.creation_modality == CreationModality.LLM_GENERATED.value
        assert row.created_from_message_id == user_message_id
        assert row.creating_model_identifier == "openai/gpt-5-mini"
        assert row.creating_model_version == "gpt-5-mini-2026-05-01"
        assert row.creating_provider == "openai"
        assert row.creating_composer_skill_hash == "sha256:composer-skill"
        assert row.creating_arguments_hash == "sha256:tool-arguments"

    @pytest.mark.parametrize(
        "blank_field",
        [
            "composer_model_identifier",
            "composer_model_version",
            "composer_provider",
            "composer_skill_hash",
            "tool_arguments_hash",
        ],
    )
    @pytest.mark.parametrize("blank_value", ["", "   ", "\t\n "])
    def test_blank_composer_provenance_fails_closed(
        self,
        tmp_path: Path,
        blank_field: str,
        blank_value: str,
    ) -> None:
        """LLM provenance fields must be non-blank, not merely non-null."""
        engine, session_id, user_message_id = _session_engine_with_user_message("Please create a CSV of the high-priority records.")
        context_kwargs = {
            "composer_model_identifier": "openai/gpt-5-mini",
            "composer_model_version": "gpt-5-mini-2026-05-01",
            "composer_provider": "openai",
            "composer_skill_hash": "sha256:composer-skill",
            "tool_arguments_hash": "sha256:tool-arguments",
        }
        context_kwargs[blank_field] = blank_value

        with pytest.raises(AuditIntegrityError, match=f"missing: {blank_field}"):
            _execute_create_blob(
                {
                    "filename": "generated.csv",
                    "mime_type": "text/csv",
                    "content": "priority,score\nhigh,99\n",
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    data_dir=str(tmp_path),
                    session_engine=engine,
                    session_id=session_id,
                    user_message_id=user_message_id,
                    user_message_content="Please create a CSV of the high-priority records.",
                    **context_kwargs,
                ),
            )

        with engine.connect() as conn:
            rows = conn.execute(select(blobs_table).where(blobs_table.c.session_id == session_id)).fetchall()
        assert rows == []

    def test_content_in_user_message_records_verbatim_without_model_identifier(self, tmp_path: Path) -> None:
        """User-supplied literal content remains verbatim even when composer provenance is present."""
        user_message_content = "Use this exact file:\nname,score\nada,42\n"
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)

        result = _execute_create_blob(
            {
                "filename": "verbatim.csv",
                "mime_type": "text/csv",
                "content": "name,score\nada,42\n",
            },
            _empty_state(),
            ToolContext(
                catalog=_mock_catalog(),
                data_dir=str(tmp_path),
                session_engine=engine,
                session_id=session_id,
                user_message_id=user_message_id,
                user_message_content=user_message_content,
                composer_model_identifier="openai/gpt-5-mini",
                composer_model_version="gpt-5-mini-2026-05-01",
                composer_provider="openai",
                composer_skill_hash="sha256:composer-skill",
                tool_arguments_hash="sha256:tool-arguments",
            ),
        )

        assert result.success is True
        with engine.connect() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == result.data["blob_id"])).one()

        assert row.creation_modality == CreationModality.VERBATIM.value
        assert row.created_from_message_id == user_message_id
        assert row.creating_model_identifier is None


# ---------------------------------------------------------------------------
# Redaction at the persistence boundary
# ---------------------------------------------------------------------------


_CANARY = "CANARY-INLINE-BLOB-CONTENT-DO-NOT-LEAK"


def test_redaction_substitutes_content_via_summarizer() -> None:
    """``content`` is replaced by the ``<inline-blob:N-bytes>`` summarizer.

    Verifies that the canary value never appears in the redacted dict
    OR its JSON serialisation — closing the same persistence-boundary
    cross-check Task 4 pinned for ``set_source``.
    """
    tel = NoopRedactionTelemetry()
    args = {
        "filename": "seed.txt",
        "mime_type": "text/plain",
        "content": _CANARY,
    }
    redacted = redact_tool_call_arguments("create_blob", args, telemetry=tel)
    # Structural keys preserved.
    assert redacted["filename"] == "seed.txt"
    assert redacted["mime_type"] == "text/plain"
    # Sensitive substitution: content is now the summarizer's str output.
    assert isinstance(redacted["content"], str)
    expected = f"<inline-blob:{len(_CANARY.encode('utf-8'))}-bytes>"
    assert redacted["content"] == expected
    # Canary MUST NOT appear anywhere in the redacted dict or its JSON form.
    serialized = json.dumps(redacted, sort_keys=True)
    assert _CANARY not in serialized, (
        "Sensitive canary value appeared in serialized output. "
        "Redaction did not remove it from the persistence path. "
        f"Serialized: {serialized!r}"
    )
    # Telemetry recorded the manifest dispatch with the type-driven shape.
    assert tel.manifest_dispatch_calls == [{"tool_name": "create_blob", "shape": "type_driven"}]


# ---------------------------------------------------------------------------
# Blob-tool summarizer type-variability (rev-2 M.10)
# ---------------------------------------------------------------------------


class TestSummarizerTypeVariability:
    """Pin rev-2 M.10: verify the summarizer is safe for every ``str`` value
    the Pydantic model admits.  Non-``str`` types (``None``, ``int``, ``bool``,
    ``list``, ``dict``) are rejected by ``content: str`` + ``extra="forbid"``
    BEFORE the summarizer runs — the lambda is never reached for them.

    Implication: the summarizer's signature can safely assume ``str`` and
    use ``len(content.encode("utf-8"))``.  This is the contract Pydantic
    enforces; this test pins the contract so a future schema change that
    relaxes ``content`` to ``str | bytes`` (or similar) fails loudly here
    rather than silently producing a wrong byte count.
    """

    def test_summarizer_safe_for_empty_string(self) -> None:
        assert _summarize_inline_blob_content("") == "<inline-blob:0-bytes>"

    def test_summarizer_safe_for_ascii_string(self) -> None:
        assert _summarize_inline_blob_content("hello") == "<inline-blob:5-bytes>"

    def test_summarizer_safe_for_non_ascii_string(self) -> None:
        # "héllo" is 6 bytes in UTF-8 (h=1, é=2, l=1, l=1, o=1).
        assert _summarize_inline_blob_content("héllo") == "<inline-blob:6-bytes>"

    def test_pydantic_model_rejects_none_before_summarizer(self) -> None:
        """``None`` never reaches the summarizer.

        ``content: str`` + ``extra="forbid"`` is Pydantic's strict-mode
        validation: a ``None`` value triggers ``ValidationError`` at
        :meth:`model_validate` time.  The test fixes this contract so the
        summarizer's safety claim (``content`` is always ``str``) is not
        a free-floating comment but a mechanically-enforced invariant.
        """
        with pytest.raises(PydanticValidationError):
            CreateBlobArgumentsModel.model_validate(
                {
                    "filename": "x.txt",
                    "mime_type": "text/plain",
                    "content": None,
                }
            )

    def test_pydantic_model_rejects_non_string_content(self) -> None:
        """Same as above for ``int`` — Pydantic strict-validates ``content: str``."""
        with pytest.raises(PydanticValidationError):
            CreateBlobArgumentsModel.model_validate(
                {
                    "filename": "x.txt",
                    "mime_type": "text/plain",
                    "content": 42,
                }
            )
