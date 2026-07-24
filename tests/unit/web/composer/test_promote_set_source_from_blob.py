"""ARG_ERROR routing + redaction for ``set_source_from_blob`` (Task 13 / Wave 2).

Final sub-task of Wave 2.  Same discipline as
``test_promote_create_blob.py`` and ``test_promote_update_blob.py``
(rev-2 BLOCKER_A).  ``set_source_from_blob`` shares the
:func:`_summarize_set_source_options` summarizer with
:class:`SetSourceArgumentsModel` (Task 4); both source-binding tools
must apply uniform redaction to caller-supplied ``options`` dicts so an
LLM that happens to include a path-like field receives the same
discipline regardless of which binding tool it invoked.
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
from sqlalchemy import insert
from sqlalchemy.pool import StaticPool

from elspeth.contracts.enums import CreationModality
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSummary
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    MANIFEST,
    SetSourceFromBlobArgumentsModel,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _execute_create_blob, _execute_patch_source_options, _execute_set_source_from_blob
from elspeth.web.composer.tools._common import ToolContext as _ToolContext
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, SOURCE_AUTHORING_KEY
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


def _option_shape_summary(*, scalar: int) -> dict[str, object]:
    return {
        "_option_shape": "mapping",
        "entry_count": scalar,
        "value_shape_counts": {
            "mapping": 0,
            "scalar": scalar,
            "sequence": 0,
            "set": 0,
        },
    }


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
    """A minimal catalog whose ``get_schema`` accepts ``text`` (the plugin
    set_source_from_blob defaults to for ``text/plain`` MIME)."""
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name=name, description=name, plugin_type="source", config_fields=[]) for name in ("csv", "json", "text")
    ]
    catalog.get_schema.return_value = {"properties": {}}
    return catalog


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


def _session_engine_with_user_message(content: str) -> tuple[Any, str, str]:
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
# Manifest shape pin
# ---------------------------------------------------------------------------


def test_set_source_from_blob_manifest_entry_is_type_driven() -> None:
    entry = MANIFEST["set_source_from_blob"]
    assert entry.argument_model is SetSourceFromBlobArgumentsModel
    assert entry.policy is None


# ---------------------------------------------------------------------------
# ARG_ERROR routing — bare ValidationError must NOT escape the handler
# ---------------------------------------------------------------------------


class TestPromoteSetSourceFromBlobArgErrorRouting:
    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """A bare ``{}`` is missing both required fields (blob_id, on_success)."""
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_source_from_blob(
                {},
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_non_dict_options_raises_tool_argument_error(self) -> None:
        """Pydantic rejects ``options: str`` before the blob lookup."""
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_source_from_blob(
                {
                    "blob_id": "anything",
                    "on_success": "out",
                    "options": "column=text",
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_on_success_raises_tool_argument_error(self) -> None:
        engine, session_id = _session_engine_with_session()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_source_from_blob(
                {"blob_id": "anything"},
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
            _execute_set_source_from_blob(
                {
                    "blob_id": "anything",
                    "on_success": "out",
                    "content": "hello",  # belongs on create_blob/update_blob
                },
                _empty_state(),
                ToolContext(
                    catalog=_mock_catalog(),
                    session_engine=engine,
                    session_id=session_id,
                ),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_placeholder_blob_id_returns_boundary_failure(self) -> None:
        """Invalid placeholder ids are rejected before blob lookup."""
        engine, session_id = _session_engine_with_session()
        result = _execute_set_source_from_blob(
            {
                "blob_id": "__missing__",
                "on_success": "out",
            },
            _empty_state(),
            ToolContext(
                catalog=_mock_catalog(),
                session_engine=engine,
                session_id=session_id,
            ),
        )

        assert result.success is False
        assert "not a valid UUID" in result.data["error"]
        assert "upload" in result.data["error"]
        assert "list_blobs" in result.data["error"]
        assert "not found" not in result.data["error"].lower()

    def test_valid_arguments_dispatch_normally(self, tmp_path: Path) -> None:
        """Functional smoke: a valid call wires the blob as the source.

        Drives the full create_blob → set_source_from_blob lifecycle so
        the post-promotion handler reaches the source-wiring path
        (versus only the validation gate).
        """
        user_message_content = "Use this exact text:\nhello"
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        catalog = _mock_catalog()

        ctx = ToolContext(
            catalog=catalog,
            data_dir=str(tmp_path),
            session_engine=engine,
            session_id=session_id,
            user_message_id=user_message_id,
            user_message_content=user_message_content,
        )
        create_result = _execute_create_blob(
            {"filename": "seed.txt", "mime_type": "text/plain", "content": "hello"},
            _empty_state(),
            ctx,
        )
        assert create_result.success is True
        blob_id = create_result.data["blob_id"]

        bind_result = _execute_set_source_from_blob(
            {
                "blob_id": blob_id,
                "on_success": "out",
                "options": {"column": "text", "schema": {"mode": "observed"}},
            },
            _empty_state(),
            ctx,
        )
        assert bind_result.success is True
        assert bind_result.updated_state.sources["source"].on_success == "out"

    def test_omitted_options_validates_at_model_layer(self) -> None:
        """``options`` is optional at the model layer (default ``{}``).

        Pin against a future refactor that changes ``options`` to
        ``Optional[dict] = None`` (which would produce ``"null"`` in
        the redacted dict — see model docstring).  The downstream
        plugin-side validation in ``_resolve_source_blob`` may still
        reject an empty ``options`` if the inferred plugin requires
        specific keys; that is the source plugin's contract, not the
        argument-model's.

        We exercise the model directly here so the test is independent
        of which source plugin is inferred from a given MIME type — that
        inference belongs to the handler's runtime path, not the
        argument-validation gate this test pins.
        """
        validated = SetSourceFromBlobArgumentsModel.model_validate({"blob_id": "anything", "on_success": "out"})
        assert validated.options == {}, (
            "Omitting options must default to {} (not None). A None default "
            "would produce 'null' in the redacted dict via the summarizer, "
            "diverging from the handler's runtime semantics where an absent "
            "options slot is treated as no caller-supplied options."
        )
        # plugin and on_validation_failure preserve None-vs-specified
        # semantics (so the handler can apply the right fallback).
        assert validated.plugin is None
        assert validated.on_validation_failure is None

    def test_llm_authored_blob_binding_stamps_source_authoring_without_unlocking_path(self, tmp_path: Path) -> None:
        """Blob-backed source provenance is stamped while path/blob_ref stay locked."""
        user_message_content = "Create generated text content for the source."
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        catalog = _mock_catalog()
        ctx = ToolContext(
            catalog=catalog,
            data_dir=str(tmp_path),
            session_engine=engine,
            session_id=session_id,
            user_message_id=user_message_id,
            user_message_content=user_message_content,
            composer_model_identifier="openai/gpt-5-mini",
            composer_model_version="gpt-5-mini-2026-05-01",
            composer_provider="openai",
            composer_skill_hash="a" * 64,
            tool_arguments_hash="b" * 64,
        )
        create_result = _execute_create_blob(
            {"filename": "generated.txt", "mime_type": "text/plain", "content": "generated row text"},
            _empty_state(),
            ctx,
        )
        assert create_result.success is True

        bind_result = _execute_set_source_from_blob(
            {
                "blob_id": create_result.data["blob_id"],
                "on_success": "out",
                "options": {"column": "text", "schema": {"mode": "observed"}},
            },
            _empty_state(),
            ctx,
        )

        assert bind_result.success is True, bind_result.data
        assert "source" in bind_result.updated_state.sources
        options = bind_result.updated_state.sources["source"].options
        assert options[SOURCE_AUTHORING_KEY] == {
            "modality": CreationModality.LLM_GENERATED.value,
            "content_hash": create_result.data["content_hash"],
            "review_event_id": None,
            "resolved_kind": None,
        }
        requirement = options[INTERPRETATION_REQUIREMENTS_KEY][0]
        assert requirement == {
            "id": "source_review:inline_source_data",
            "kind": "invented_source",
            "user_term": "inline_source_data",
            "status": "pending",
            "draft": "generated row text",
            "event_id": None,
            "accepted_value": None,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": None,
        }

        forged_authoring_patch = _execute_patch_source_options(
            {
                "patch": {
                    SOURCE_AUTHORING_KEY: {
                        "modality": CreationModality.VERBATIM.value,
                        "content_hash": "0" * 64,
                        "review_event_id": "forged-review",
                        "resolved_kind": "forged-kind",
                    }
                }
            },
            bind_result.updated_state,
            ctx,
        )
        assert forged_authoring_patch.success is False
        assert SOURCE_AUTHORING_KEY in forged_authoring_patch.data["error"]

        patch_result = _execute_patch_source_options(
            {"patch": {"path": str(tmp_path / "other.txt")}},
            bind_result.updated_state,
            ctx,
        )
        assert patch_result.success is False
        assert "Cannot patch" in patch_result.data["error"]


# ---------------------------------------------------------------------------
# Redaction at the persistence boundary
# ---------------------------------------------------------------------------


_CANARY = "CANARY-SET-SOURCE-FROM-BLOB-PATH-DO-NOT-LEAK"


def test_redaction_substitutes_options_via_summarizer() -> None:
    """``options`` is replaced by the canonical-JSON shape summary.

    Uniformity-with-set_source contract: both source-binding tools share
    the same summarizer.  The summary must preserve shape without echoing
    plugin option values such as paths, blob refs, or credentials.
    """
    tel = NoopRedactionTelemetry()
    args = {
        "blob_id": "some-blob-id",
        "on_success": "out",
        "options": {"path": _CANARY, "blob_ref": "abc123"},
    }
    redacted = redact_tool_call_arguments("set_source_from_blob", args, telemetry=tel)
    # Structural keys preserved.
    assert redacted["blob_id"] == "some-blob-id"
    assert redacted["on_success"] == "out"
    # options is now the summarizer's str output.
    assert isinstance(redacted["options"], str)
    assert json.loads(redacted["options"]) == _option_shape_summary(scalar=2)
    # The canary value MUST NOT appear in the redacted dict OR its JSON form.
    serialized = json.dumps(redacted, sort_keys=True)
    assert _CANARY not in serialized
    # Telemetry recorded the manifest dispatch with the type-driven shape.
    assert tel.manifest_dispatch_calls == [{"tool_name": "set_source_from_blob", "shape": "type_driven"}]


def test_redaction_hides_paths_without_blob_ref() -> None:
    """Path values are redacted even when no blob_ref marker is present."""
    tel = NoopRedactionTelemetry()
    args = {
        "blob_id": "id",
        "on_success": "out",
        "options": {"path": "/tmp/data.csv"},
    }
    redacted = redact_tool_call_arguments("set_source_from_blob", args, telemetry=tel)
    assert isinstance(redacted["options"], str)
    assert json.loads(redacted["options"]) == _option_shape_summary(scalar=1)
    assert "/tmp/data.csv" not in redacted["options"]


# ---------------------------------------------------------------------------
# TSV-delimiter parity (bug elspeth-da09ed23d4)
# ---------------------------------------------------------------------------


class TestSetSourceFromBlobTsvDelimiter:
    """A ``.tsv`` blob (uploaded as ``text/csv``) must bind a csv source whose
    ``delimiter`` is a tab, matching what ``inspect_blob_content`` reports.

    Without this, ``CSVSourceConfig.delimiter`` defaults to comma and the
    tab-separated rows parse as a single column at runtime — the inspect-vs-bind
    parity gap the ticket names.
    """

    def _bind_csv_blob(
        self,
        *,
        filename: str,
        content: str,
        tmp_path: Path,
        options: dict[str, Any] | None = None,
    ) -> Any:
        # Embed the blob content verbatim in the user message so the blob is
        # classified VERBATIM (operator-supplied), not LLM-authored — that keeps
        # the harness free of full composer provenance just to exercise the
        # delimiter-derivation path.
        user_message_content = f"Bind this tabular blob as the source:\n{content}"
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        catalog = _mock_catalog()
        ctx = ToolContext(
            catalog=catalog,
            data_dir=str(tmp_path),
            session_engine=engine,
            session_id=session_id,
            user_message_id=user_message_id,
            user_message_content=user_message_content,
        )
        create_result = _execute_create_blob(
            {"filename": filename, "mime_type": "text/csv", "content": content},
            _empty_state(),
            ctx,
        )
        assert create_result.success is True, create_result.data
        bind_result = _execute_set_source_from_blob(
            {
                "blob_id": create_result.data["blob_id"],
                "on_success": "out",
                "options": options if options is not None else {"schema": {"mode": "observed"}},
            },
            _empty_state(),
            ctx,
        )
        return bind_result

    def test_tsv_blob_binds_csv_source_with_tab_delimiter(self, tmp_path: Path) -> None:
        bind_result = self._bind_csv_blob(
            filename="data.tsv",
            content="a\tb\tc\n1\t2\t3\n",
            tmp_path=tmp_path,
        )
        assert bind_result.success is True, bind_result.data
        source = bind_result.updated_state.sources["source"]
        assert source.plugin == "csv"
        assert source.options.get("delimiter") == "\t"

    def test_caller_supplied_delimiter_is_not_overridden(self, tmp_path: Path) -> None:
        bind_result = self._bind_csv_blob(
            filename="data.tsv",
            content="a;b;c\n1;2;3\n",
            tmp_path=tmp_path,
            options={"delimiter": ";", "schema": {"mode": "observed"}},
        )
        assert bind_result.success is True, bind_result.data
        source = bind_result.updated_state.sources["source"]
        assert source.options.get("delimiter") == ";"

    def test_csv_blob_does_not_inject_delimiter(self, tmp_path: Path) -> None:
        bind_result = self._bind_csv_blob(
            filename="data.csv",
            content="a,b,c\n1,2,3\n",
            tmp_path=tmp_path,
        )
        assert bind_result.success is True, bind_result.data
        source = bind_result.updated_state.sources["source"]
        assert source.plugin == "csv"
        assert source.options.get("delimiter") is None
