"""ARG_ERROR routing + redaction for ``set_pipeline`` (Task 14 / Wave 3).

Wave 3 sub-task 1/2.  Same discipline as ``test_promote_set_source.py``
(Task 4) and the three Wave 2 promotion tests
(``test_promote_create_blob.py``, ``test_promote_update_blob.py``,
``test_promote_set_source_from_blob.py``).  Rev-2 BLOCKER_A applies:
promoted handlers MUST catch :class:`pydantic.ValidationError` and
re-raise as :class:`ToolArgumentError`.  A bare ``ValidationError``
escaping the handler hits ``service.py:2564`` (→
:class:`ComposerPluginCrashError` → HTTP 500) — wrong disposition for
Tier-3 input.

Tests pin:
  * manifest shape (type-driven),
  * exception-class + ``__cause__`` chain on invalid arguments,
  * ``extra="forbid"`` at every nested level,
  * conditional inline_blob inner-fields (Pydantic structural enforcement
    replaces the deleted ``elspeth-4e79436719 §Bug A`` walker guard),
  * valid dispatch produces a working pipeline (functional smoke),
  * :func:`redact_tool_call_arguments` collapses every Sensitive surface
    (``source.options``, ``source.inline_blob.content``,
    ``nodes[*].options``, ``outputs[*].options``) via the declared
    summarizers, and passes structurally-typed fields
    (``nodes[*].routes``: ``dict[str, str]``; ``nodes[*].trigger``: typed
    sub-model) through verbatim (F3 — see
    ``docs/composer/evidence/composer-phase-2-followup-prompt-F1-F6.md``).
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
    SetPipelineArgumentsModel,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _execute_create_blob, _execute_set_pipeline
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, SOURCE_AUTHORING_KEY
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
    """Minimal catalog accepting the ``text`` source plugin used by the
    functional smoke test below.  A bare ``MagicMock`` is sufficient — the
    paths exercised here do not consult the catalog's schema registry."""
    catalog = MagicMock(spec=CatalogService)
    catalog.get_schema.return_value = {"properties": {}}
    return catalog


def _session_engine_with_session() -> tuple[Any, str]:
    """Minimal session DB with one row, matching the Wave 2 helpers."""
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


def _minimal_valid_args() -> dict[str, Any]:
    """The smallest dict that passes :class:`SetPipelineArgumentsModel`.

    Top-level ``required: ["source", "nodes", "edges", "outputs"]``; the
    nested ``source`` requires ``plugin`` + ``on_success``.  Empty lists
    are valid arrays per JSON schema and Pydantic.  ``nodes`` is empty so
    the handler skips per-node validation paths.  ``outputs`` is empty so
    the handler skips per-output validation paths.
    """
    return {
        "source": {"plugin": "text", "on_success": "rows"},
        "nodes": [],
        "edges": [],
        "outputs": [],
    }


# ---------------------------------------------------------------------------
# Manifest shape pin
# ---------------------------------------------------------------------------


def test_set_pipeline_manifest_entry_is_type_driven() -> None:
    entry = MANIFEST["set_pipeline"]
    assert entry.argument_model is SetPipelineArgumentsModel
    assert entry.policy is None


# ---------------------------------------------------------------------------
# ARG_ERROR routing — bare ValidationError must NOT escape the handler
# ---------------------------------------------------------------------------


class TestPromoteSetPipelineArgErrorRouting:
    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """A bare ``{}`` is missing all four top-level required fields."""
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline({}, _empty_state(), ToolContext(catalog=_mock_catalog()))
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_source_plugin_raises_tool_argument_error(self) -> None:
        """Top-level source.plugin omission is a structured Pydantic failure."""
        args = _minimal_valid_args()
        del args["source"]["plugin"]
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        cause = exc_info.value.__cause__
        assert isinstance(cause, PydanticValidationError)
        # The structured path is preserved on __cause__ for audit, NOT
        # echoed in args[0] (compose-loop leak-prevention discipline).
        assert any(err["loc"] == ("source", "plugin") for err in cause.errors())

    def test_extra_top_level_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' at the top level rejects misrouted argument shapes.

        ``filename`` belongs on ``create_blob``/``update_blob``; supplying
        it at the top level of a ``set_pipeline`` call signals the LLM
        targeted the wrong tool.
        """
        args = _minimal_valid_args()
        args["filename"] = "stray.csv"
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_on_source_raises_tool_argument_error(self) -> None:
        """extra='forbid' on nested ``_SetPipelineSourceModel``.

        ``label`` does not appear on the source schema.
        """
        args = _minimal_valid_args()
        args["source"]["label"] = "Source A"
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_partial_inline_blob_raises_with_nested_path(self) -> None:
        """inline_blob present but incomplete surfaces nested required fields.

        Pydantic only walks ``_InlineBlobModel`` when ``source.inline_blob``
        is supplied; the nested required-fields are then structurally
        enforced.  This is the elspeth-4e79436719 §Bug A invariant
        ("conditional inline_blob inner fields"), now expressed via the
        Pydantic optional-Model layout rather than the deleted
        ``_TOOL_REQUIRED_PATHS`` walker's optional_ancestor mechanism.
        """
        args = _minimal_valid_args()
        args["source"]["inline_blob"] = {"filename": "data.csv"}  # missing mime_type + content
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        cause = exc_info.value.__cause__
        assert isinstance(cause, PydanticValidationError)
        missing_locs = {err["loc"] for err in cause.errors() if err["type"] == "missing"}
        assert ("source", "inline_blob", "mime_type") in missing_locs
        assert ("source", "inline_blob", "content") in missing_locs
        # ``filename`` WAS supplied — must not appear in the missing-set.
        assert ("source", "inline_blob", "filename") not in missing_locs

    def test_inline_blob_extra_field_rejected(self) -> None:
        """extra='forbid' on nested ``_InlineBlobModel``."""
        args = _minimal_valid_args()
        args["source"]["inline_blob"] = {
            "filename": "data.csv",
            "mime_type": "text/csv",
            "content": "a,b,c",
            "extra_field": "stray",
        }
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_node_missing_required_input_raises(self) -> None:
        """nodes[*] inner ``required: [id, node_type, input]`` is enforced."""
        args = _minimal_valid_args()
        args["nodes"] = [{"id": "t1", "node_type": "transform"}]  # missing input
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        cause = exc_info.value.__cause__
        assert isinstance(cause, PydanticValidationError)
        assert any(err["loc"] == ("nodes", 0, "input") for err in cause.errors())

    def test_edge_missing_required_to_node_raises(self) -> None:
        """edges[*] inner ``required: [id, from_node, to_node, edge_type]``."""
        args = _minimal_valid_args()
        args["edges"] = [{"id": "e1", "from_node": "a", "edge_type": "on_success"}]
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        cause = exc_info.value.__cause__
        assert isinstance(cause, PydanticValidationError)
        assert any(err["loc"] == ("edges", 0, "to_node") for err in cause.errors())

    def test_output_missing_required_plugin_raises(self) -> None:
        """outputs[*] inner ``required: [sink_name, plugin]``."""
        args = _minimal_valid_args()
        args["outputs"] = [{"sink_name": "main"}]
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))
        cause = exc_info.value.__cause__
        assert isinstance(cause, PydanticValidationError)
        assert any(err["loc"] == ("outputs", 0, "plugin") for err in cause.errors())

    def test_valid_arguments_dispatch_normally(self, tmp_path: Path) -> None:
        """Functional smoke: a valid inline_blob set_pipeline materialises
        the blob and binds it as the source.

        Drives the full inline_blob materialisation path so the
        post-promotion handler reaches the source-wiring + blob-persistence
        seams (versus only the validation gate).  This also exercises the
        ``_prepare_blob_create`` second-caller pathway whose dead
        isinstance guards were removed in this same commit.
        """
        user_message_content = "Use this exact text file:\nhello"
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        catalog = _mock_catalog()
        output_path = tmp_path / "outputs" / "out.csv"

        args = {
            "source": {
                "plugin": "text",
                "on_success": "rows",
                "options": {
                    "column": "text",
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                "inline_blob": {
                    "filename": "input.txt",
                    "mime_type": "text/plain",
                    "content": "hello",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [
                {
                    "sink_name": "rows",
                    "plugin": "csv",
                    "options": {
                        "path": str(output_path),
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
        }

        result = _execute_set_pipeline(
            args,
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
        assert result.success is True
        source = result.updated_state.sources["source"]
        assert source.plugin == "text"
        # The handler resolves blob_ref into source.options authoritatively.
        assert "blob_ref" in source.options
        # Inline content must not leak into the affected/data summary.
        assert "hello" not in str(result.to_dict())

    def test_inline_blob_without_message_or_composer_provenance_fails_closed(self, tmp_path: Path) -> None:
        """Inline source blobs must not silently persist as verbatim without provenance."""
        engine, session_id = _session_engine_with_session()
        args = {
            "source": {
                "plugin": "text",
                "on_success": "rows",
                "options": {
                    "column": "text",
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                "inline_blob": {
                    "filename": "input.txt",
                    "mime_type": "text/plain",
                    "content": "generated row",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        with pytest.raises(AuditIntegrityError, match="missing: user_message_id"):
            _execute_set_pipeline(
                args,
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
    def test_inline_blob_contained_content_without_message_id_fails_closed(
        self,
        tmp_path: Path,
        missing_user_message_id: str | None,
    ) -> None:
        """Verbatim inline blobs require both message content and message id."""
        engine, session_id = _session_engine_with_session()
        user_message_content = "Use this exact source text:\ngenerated row"
        args = {
            "source": {
                "plugin": "text",
                "on_success": "rows",
                "options": {
                    "column": "text",
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                "inline_blob": {
                    "filename": "input.txt",
                    "mime_type": "text/plain",
                    "content": "generated row",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        with pytest.raises(AuditIntegrityError, match="missing: user_message_id"):
            _execute_set_pipeline(
                args,
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

    def test_non_blob_source_rejects_manual_source_authoring(self) -> None:
        """source_authoring is reserved for blob provenance, not caller options."""
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {
                    "path": "/data/in.csv",
                    "schema": {"mode": "observed"},
                    SOURCE_AUTHORING_KEY: {
                        "modality": CreationModality.LLM_GENERATED.value,
                        "content_hash": "0" * 64,
                        "review_event_id": None,
                        "resolved_kind": None,
                    },
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(
            args,
            _empty_state(),
            ToolContext(catalog=_mock_catalog()),
        )

        assert result.success is False
        assert "source" not in result.updated_state.sources
        assert SOURCE_AUTHORING_KEY in result.data["error"]

    def test_csv_fixed_schema_accepts_advertised_field_definition_shape(self, tmp_path: Path) -> None:
        """CSV prevalidation accepts the field shape exposed by plugin JSON Schema."""
        user_message_content = "Use this exact CSV:\nurl\nhttps://example.test\n"
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        output_path = tmp_path / "outputs" / "out.csv"

        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {
                    "schema": {
                        "mode": "fixed",
                        "fields": [{"name": "url", "field_type": "str"}],
                    }
                },
                "inline_blob": {
                    "filename": "input.csv",
                    "mime_type": "text/csv",
                    "content": "url\nhttps://example.test\n",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [
                {
                    "sink_name": "rows",
                    "plugin": "csv",
                    "options": {
                        "path": str(output_path),
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
        }

        result = _execute_set_pipeline(
            args,
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

        assert result.success is True, result.data
        assert result.validation.is_valid is True
        source = result.updated_state.sources["source"]
        assert source.plugin == "csv"
        assert source.options["schema"]["fields"] == ({"name": "url", "field_type": "str"},)

    def test_inline_blob_llm_authored_source_records_authoring_metadata(self, tmp_path: Path) -> None:
        """LLM-authored inline source blobs must stamp source-level provenance."""
        user_message_content = "Create a tiny generated CSV for the pipeline."
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"schema": {"mode": "observed"}},
                "inline_blob": {
                    "filename": "generated.csv",
                    "mime_type": "text/csv",
                    "content": "name,score\nada,42\n",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(
            args,
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

        assert result.success is True, result.data
        assert "source" in result.updated_state.sources
        options = result.updated_state.sources["source"].options
        assert options["blob_ref"] == result.data["inline_blob"]["blob_id"]
        assert SOURCE_AUTHORING_KEY in options
        assert options[SOURCE_AUTHORING_KEY] == {
            "modality": CreationModality.LLM_GENERATED.value,
            "content_hash": result.data["inline_blob"]["content_hash"],
            "review_event_id": None,
            "resolved_kind": None,
        }
        requirements = options[INTERPRETATION_REQUIREMENTS_KEY]
        assert len(requirements) == 1
        assert requirements[0] == {
            "id": "source_review:inline_source_data",
            "kind": "invented_source",
            "user_term": "inline_source_data",
            "status": "pending",
            "draft": "name,score\nada,42\n",
            "event_id": None,
            "accepted_value": None,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": None,
        }
        with engine.connect() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == options["blob_ref"])).one()
        assert row.creation_modality == CreationModality.LLM_GENERATED.value

    def test_inline_blob_llm_authored_url_source_records_url_list_review_requirement(self, tmp_path: Path) -> None:
        """Headered URL CSVs get the tutorial-stable invented-source user_term."""
        user_message_content = "Create a generated URL CSV for the pipeline."
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        url_csv = "url\nhttps://example.test\nhttps://example.gov.au\n"
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"schema": {"mode": "observed"}},
                "inline_blob": {
                    "filename": "generated.csv",
                    "mime_type": "text/csv",
                    "content": url_csv,
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(
            args,
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

        assert result.success is True, result.data
        assert "source" in result.updated_state.sources
        requirement = result.updated_state.sources["source"].options[INTERPRETATION_REQUIREMENTS_KEY][0]
        assert requirement["kind"] == "invented_source"
        assert requirement["user_term"] == "inline_source_url_list"
        assert requirement["draft"] == url_csv

    def test_set_pipeline_rejects_unreviewed_drop_of_web_scrape_raw_fields(self) -> None:
        """Dropping web-scrape raw fields is a user-visible pipeline decision."""
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "/data/urls.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "fetch_pages",
                    "node_type": "transform",
                    "plugin": "web_scrape",
                    "input": "rows",
                    "on_success": "scraped_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                        "http": {
                            "abuse_contact": "noreply@dta.gov.au",
                            "scraping_reason": "DTA technical demonstration",
                            "allowed_hosts": "public_only",
                        },
                    },
                },
                {
                    "id": "drop_raw_html",
                    "node_type": "transform",
                    "plugin": "field_mapper",
                    "input": "scraped_rows",
                    "on_success": "clean_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "mapping": {"url": "url"},
                        "select_only": True,
                    },
                },
            ],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))

        assert result.success is False
        assert "drop_raw_html_fields" in result.data["error"]
        assert "pipeline_decision" in result.data["error"]

    def test_set_pipeline_rejects_cleanup_named_mapper_that_preserves_web_scrape_raw_fields(self) -> None:
        """A node named as cleanup cannot preserve the exact raw fields it claims to drop."""
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "/data/urls.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "fetch_pages",
                    "node_type": "transform",
                    "plugin": "web_scrape",
                    "input": "rows",
                    "on_success": "scraped_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                        "http": {
                            "abuse_contact": "noreply@dta.gov.au",
                            "scraping_reason": "DTA technical demonstration",
                            "allowed_hosts": "public_only",
                        },
                    },
                },
                {
                    "id": "drop_raw_html_fields",
                    "node_type": "transform",
                    "plugin": "field_mapper",
                    "input": "scraped_rows",
                    "on_success": "clean_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "mapping": {
                            "url": "url",
                            "content": "content",
                        },
                        "select_only": True,
                    },
                },
            ],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))

        assert result.success is False
        assert "preserves web-scrape raw field" in result.data["error"]
        assert "content" in result.data["error"]

    def test_set_pipeline_rejects_malformed_interpretation_requirements_without_crashing(self) -> None:
        """Malformed review metadata is Tier-3 tool input and must be a clean rejection."""
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "/data/urls.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "fetch_pages",
                    "node_type": "transform",
                    "plugin": "web_scrape",
                    "input": "rows",
                    "on_success": "scraped_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "url_field": "url",
                        "content_field": "html",
                        "fingerprint_field": "page_fingerprint",
                        "http": {
                            "abuse_contact": "noreply@dta.gov.au",
                            "scraping_reason": "DTA technical demonstration",
                            "allowed_hosts": "public_only",
                        },
                    },
                },
                {
                    "id": "drop_raw_html",
                    "node_type": "transform",
                    "plugin": "field_mapper",
                    "input": "scraped_rows",
                    "on_success": "clean_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "mapping": {"url": "url"},
                        "select_only": True,
                        INTERPRETATION_REQUIREMENTS_KEY: {
                            "id": "drop_raw_html_review",
                            "kind": "pipeline_decision",
                            "user_term": "drop_raw_html_fields",
                            "status": "pending",
                            "draft": "Drop the scraped raw HTML and fingerprint fields before saving the JSON output.",
                        },
                    },
                },
            ],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))

        assert result.success is False
        assert "interpretation_requirements must be a list" in result.data["error"]

    def test_set_pipeline_rejects_raw_cleanup_review_on_llm_node(self) -> None:
        """A raw-cleanup review must be attached to the field_mapper doing the cleanup."""
        cleanup_requirement = {
            "id": "drop_raw_html_review",
            "kind": "pipeline_decision",
            "user_term": "drop_raw_html_fields",
            "status": "pending",
            "draft": "Drop the scraped raw HTML and fingerprint fields before saving the JSON output.",
        }
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "/data/urls.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "fetch_pages",
                    "node_type": "transform",
                    "plugin": "web_scrape",
                    "input": "rows",
                    "on_success": "scraped_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                        "http": {
                            "abuse_contact": "noreply@dta.gov.au",
                            "scraping_reason": "DTA technical demonstration",
                            "allowed_hosts": "public_only",
                        },
                    },
                },
                {
                    "id": "identify_primary_colours",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "scraped_rows",
                    "on_success": "coloured_rows",
                    "on_error": "discard",
                    "options": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-haiku-4.5",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                        "prompt_template": "Read {{ row.content }}.",
                        "schema": {"mode": "observed"},
                        INTERPRETATION_REQUIREMENTS_KEY: [cleanup_requirement],
                    },
                },
                {
                    "id": "drop_raw_html",
                    "node_type": "transform",
                    "plugin": "field_mapper",
                    "input": "coloured_rows",
                    "on_success": "clean_rows",
                    "on_error": "discard",
                    "options": {
                        "schema": {"mode": "observed"},
                        "mapping": {"url": "url", "primary_colour_result": "primary_colour_result"},
                        "select_only": True,
                        INTERPRETATION_REQUIREMENTS_KEY: [cleanup_requirement],
                    },
                },
            ],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))

        assert result.success is False
        assert "identify_primary_colours" in result.data["error"]
        assert "must be implemented by a field_mapper" in result.data["error"]

    def test_existing_llm_blob_url_source_records_url_list_review_requirement(self, tmp_path: Path) -> None:
        """source.blob_id preserves the same source-review gate as inline_blob."""
        user_message_content = "Create a generated URL CSV for the pipeline."
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        url_csv = "url\nhttps://example.test\nhttps://example.gov.au\n"
        context = ToolContext(
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
        )
        create_result = _execute_create_blob(
            {
                "filename": "generated.csv",
                "mime_type": "text/csv",
                "content": url_csv,
            },
            _empty_state(),
            context,
        )
        assert create_result.success is True, create_result.data
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"schema": {"mode": "observed"}},
                "blob_id": create_result.data["blob_id"],
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(args, _empty_state(), context)

        assert result.success is True, result.data
        assert "source" in result.updated_state.sources
        options = result.updated_state.sources["source"].options
        assert options[SOURCE_AUTHORING_KEY] == {
            "modality": CreationModality.LLM_GENERATED.value,
            "content_hash": create_result.data["content_hash"],
            "review_event_id": None,
            "resolved_kind": None,
        }
        requirement = options[INTERPRETATION_REQUIREMENTS_KEY][0]
        assert requirement["kind"] == "invented_source"
        assert requirement["user_term"] == "inline_source_url_list"
        assert requirement["draft"] == url_csv

    def test_inline_blob_llm_authored_source_prevalidation_ignores_review_metadata(self, tmp_path: Path) -> None:
        """Plugin prevalidation strips web-only interpretation metadata but preserves it in state."""
        user_message_content = "Create a generated CSV for later source review."
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {
                    "schema": {"mode": "observed"},
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "source_review",
                            "kind": "invented_source",
                            "user_term": "inline_source_url_list",
                            "status": "pending",
                            "draft": "url\nhttps://example.test\n",
                            "event_id": None,
                            "accepted_value": None,
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": None,
                        }
                    ],
                },
                "inline_blob": {
                    "filename": "generated.csv",
                    "mime_type": "text/csv",
                    "content": "url\nhttps://example.test\n",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(
            args,
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

        assert result.success is True, result.data
        assert "source" in result.updated_state.sources
        options = result.updated_state.sources["source"].options
        assert INTERPRETATION_REQUIREMENTS_KEY in options
        assert SOURCE_AUTHORING_KEY in options
        assert len(options[INTERPRETATION_REQUIREMENTS_KEY]) == 1
        assert options[INTERPRETATION_REQUIREMENTS_KEY][0]["user_term"] == "inline_source_url_list"

    def test_llm_node_records_pending_prompt_template_review_requirement(self) -> None:
        """Every LLM prompt template authored by set_pipeline carries a Class 3 gate."""
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "summarise",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "rows",
                    "on_success": "out",
                    "options": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-haiku-4.5",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                        "prompt_template": "Summarise {{ row.text }}.",
                        "schema": {"mode": "observed"},
                    },
                }
            ],
            "edges": [],
            "outputs": [],
        }

        result = _execute_set_pipeline(args, _empty_state(), ToolContext(catalog=_mock_catalog()))

        assert result.success is True, result.data
        node = result.updated_state.nodes[0]
        requirements = node.options[INTERPRETATION_REQUIREMENTS_KEY]
        # set_pipeline routes node options through the composite LLM-review
        # auto-stager — every LLM node with a non-empty ``model`` and
        # ``prompt_template`` acquires both default gates in the same
        # mutation.
        assert len(requirements) == 2
        assert requirements[0] == {
            "id": "prompt_template_review:summarise",
            "kind": "llm_prompt_template",
            "user_term": "llm_prompt_template:summarise",
            "status": "pending",
            "draft": "Summarise {{ row.text }}.",
            "event_id": None,
            "accepted_value": None,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": None,
        }
        assert requirements[1] == {
            "id": "model_choice_review:summarise",
            "kind": "llm_model_choice",
            "user_term": "llm_model_choice:summarise",
            "status": "pending",
            "draft": "anthropic/claude-haiku-4.5",
            "event_id": None,
            "accepted_value": None,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": None,
        }

    def test_omitted_metadata_validates_at_model_layer(self) -> None:
        """``metadata`` is optional at the top level; absent leaves the field None."""
        validated = SetPipelineArgumentsModel.model_validate(_minimal_valid_args())
        assert validated.metadata is None

    def test_omitted_source_options_defaults_to_empty_dict(self) -> None:
        """Mirrors :class:`SetSourceFromBlobArgumentsModel.options` semantics.

        Pin against a future refactor changing ``options`` to
        ``Optional[dict] = None`` (which would produce ``"null"`` in the
        redacted view via the summarizer — divergent from the handler's
        absent-equals-empty runtime semantics).
        """
        validated = SetPipelineArgumentsModel.model_validate(_minimal_valid_args())
        assert validated.source.options == {}


# ---------------------------------------------------------------------------
# Redaction at the persistence boundary
# ---------------------------------------------------------------------------


_CANARY_PATH = "CANARY-SET-PIPELINE-SOURCE-PATH-DO-NOT-LEAK"
_CANARY_NODE_OPT = "CANARY-NODE-OPTIONS-DO-NOT-LEAK"
_CANARY_OUTPUT_OPT = "CANARY-OUTPUT-OPTIONS-DO-NOT-LEAK"
_CANARY_INLINE = "CANARY-INLINE-BLOB-CONTENT-DO-NOT-LEAK"
_CANARY_ROUTES = "CANARY-NODE-ROUTES-DO-NOT-LEAK"
_CANARY_TRIGGER = "CANARY-NODE-TRIGGER-DO-NOT-LEAK"


def test_redaction_substitutes_source_options_via_summarizer() -> None:
    """``source.options`` is replaced by the canonical-JSON shape summary
    (:func:`_summarize_set_source_options`).

    Uniformity-with-set_source contract: ``set_pipeline`` shares the
    summarizer with ``set_source`` and ``set_source_from_blob``. The summary
    preserves option shape but replaces scalar values before JSON encoding.
    """
    tel = NoopRedactionTelemetry()
    args = {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": _CANARY_PATH, "blob_ref": "abc123"},
        },
        "nodes": [],
        "edges": [],
        "outputs": [],
    }
    redacted = redact_tool_call_arguments("set_pipeline", args, telemetry=tel)
    # Structural keys preserved at the top.
    assert redacted["source"]["plugin"] == "csv"
    assert redacted["source"]["on_success"] == "rows"
    # options collapses to the summarizer's str output.
    assert isinstance(redacted["source"]["options"], str)
    assert json.loads(redacted["source"]["options"]) == {
        "blob_ref": "<redacted-option-value>",
        "path": "<redacted-option-value>",
    }
    # The canary path MUST NOT appear anywhere in the redacted output.
    serialized = json.dumps(redacted, sort_keys=True)
    assert _CANARY_PATH not in serialized
    # Telemetry recorded the manifest dispatch with the type-driven shape.
    assert tel.manifest_dispatch_calls == [{"tool_name": "set_pipeline", "shape": "type_driven"}]


def test_redaction_substitutes_inline_blob_content_via_summarizer() -> None:
    """``source.inline_blob.content`` is replaced by ``<inline-blob:N-bytes>``."""
    tel = NoopRedactionTelemetry()
    args = {
        "source": {
            "plugin": "text",
            "on_success": "rows",
            "inline_blob": {
                "filename": "input.txt",
                "mime_type": "text/plain",
                "content": _CANARY_INLINE,
            },
        },
        "nodes": [],
        "edges": [],
        "outputs": [],
    }
    redacted = redact_tool_call_arguments("set_pipeline", args, telemetry=tel)
    inline = redacted["source"]["inline_blob"]
    # filename + mime_type pass through verbatim.
    assert inline["filename"] == "input.txt"
    assert inline["mime_type"] == "text/plain"
    # content collapses to the byte-length summary scalar.
    assert isinstance(inline["content"], str)
    assert inline["content"].startswith("<inline-blob:")
    assert inline["content"].endswith("-bytes>")
    # The canary content MUST NOT appear anywhere.
    serialized = json.dumps(redacted, sort_keys=True)
    assert _CANARY_INLINE not in serialized


def test_redaction_substitutes_nested_node_and_output_dicts() -> None:
    """``nodes[*].options`` and ``outputs[*].options`` are collapsed by the
    shared summarizer; ``nodes[*].routes`` and ``nodes[*].trigger`` pass
    through verbatim under their structural typings (F3).

    Two channels for two surfaces:

    *  ``options`` (both node and output) is typed ``dict[str, Any]`` and
       carries a Sensitive marker — the §4.4.2 adequacy guard's
       inspection-resistant case.  The summarizer collapses the dict to
       a JSON-string preview.

    *  ``routes`` is ``dict[str, str]`` (route-label → sink/connection
       identifier) and ``trigger`` is a typed :class:`_NodeTriggerModel`
       — the §4.4.2 structural-exemption case.  No Sensitive marker is
       needed because the walker descends to closed-list scalars
       (``str``, ``int|None``, ``float|None``, ``str|None``).  The
       redactor returns these fields verbatim — that's the F3 contract.
    """
    tel = NoopRedactionTelemetry()
    # ``_CANARY_TRIGGER`` is parked on ``trigger.condition`` (the only
    # ``str`` slot in the typed sub-model) to give the regression check a
    # value-bearing leaf.  Putting it on ``count`` would require an int
    # canary; ``condition`` keeps the canary as a string.
    args = {
        "source": {"plugin": "csv", "on_success": "rows"},
        "nodes": [
            {
                "id": "n1",
                "node_type": "transform",
                "input": "rows",
                "options": {"prompt_template": _CANARY_NODE_OPT},
                "routes": {"true": _CANARY_ROUTES},
                "trigger": {"condition": _CANARY_TRIGGER},
            }
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": _CANARY_OUTPUT_OPT},
            }
        ],
    }
    redacted = redact_tool_call_arguments("set_pipeline", args, telemetry=tel)
    # ``options`` (Sensitive ``dict[str, Any]``) collapses to a str — the
    # summarizer's canonical-JSON shape output.
    assert isinstance(redacted["nodes"][0]["options"], str)
    assert isinstance(redacted["outputs"][0]["options"], str)
    assert json.loads(redacted["nodes"][0]["options"]) == {"prompt_template": "<redacted-option-value>"}
    assert json.loads(redacted["outputs"][0]["options"]) == {"path": "<redacted-option-value>"}
    # ``routes`` and ``trigger`` pass through with their original shapes
    # — structurally exempt under §4.4.2 (closed-list scalar element types).
    assert redacted["nodes"][0]["routes"] == {"true": _CANARY_ROUTES}
    # ``trigger`` is dumped from :class:`_NodeTriggerModel` via the
    # redaction walker's BaseModel descent — the redacted shape carries
    # every declared field (the absent ``count`` / ``timeout_seconds``
    # slots surface as ``None``, matching the model defaults).  The
    # canary lives on ``condition``; the other slots are present but
    # null.
    assert redacted["nodes"][0]["trigger"] == {
        "condition": _CANARY_TRIGGER,
        "count": None,
        "timeout_seconds": None,
    }
    # Option canaries are removed by the shared option summarizer. Routes and
    # triggers remain typed Python containers and keep their structural scalar
    # values under the F3 contract.
    serialized = json.dumps(redacted, sort_keys=True)
    assert _CANARY_NODE_OPT not in serialized
    assert _CANARY_OUTPUT_OPT not in serialized
    assert serialized.count(_CANARY_ROUTES) == 1
    assert serialized.count(_CANARY_TRIGGER) == 1


# ---------------------------------------------------------------------------
# TSV-delimiter parity on the inline_blob bind path (bug elspeth-da09ed23d4)
# ---------------------------------------------------------------------------


class TestSetPipelineInlineBlobTsvDelimiter:
    """The inline_blob bind branch must derive a tab delimiter for a ``.tsv``
    inline blob, mirroring ``set_source_from_blob`` and ``inspect_blob_content``.

    ``_MIME_TO_SOURCE`` is mime-keyed only, so without the filename-derived
    delimiter a tab-separated inline blob binds the csv plugin with the comma
    default and parses as one column at runtime.
    """

    def _set_pipeline_with_inline_csv_blob(
        self,
        *,
        filename: str,
        content: str,
        tmp_path: Path,
        extra_source_options: dict[str, Any] | None = None,
    ) -> Any:
        # Embed the blob content verbatim in the user message so the inline
        # blob is classified VERBATIM (operator-supplied), avoiding the
        # LLM-authored provenance requirement for this delimiter-path test.
        user_message_content = f"Use this tabular file:\n{content}"
        engine, session_id, user_message_id = _session_engine_with_user_message(user_message_content)
        catalog = _mock_catalog()
        source_options: dict[str, Any] = {"schema": {"mode": "observed"}}
        if extra_source_options:
            source_options.update(extra_source_options)
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": source_options,
                "inline_blob": {
                    "filename": filename,
                    "mime_type": "text/csv",
                    "content": content,
                },
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [
                {
                    "sink_name": "rows",
                    "plugin": "csv",
                    "options": {
                        "path": str(tmp_path / "outputs" / "out.csv"),
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
        }
        return _execute_set_pipeline(
            args,
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

    def test_tsv_inline_blob_binds_csv_source_with_tab_delimiter(self, tmp_path: Path) -> None:
        result = self._set_pipeline_with_inline_csv_blob(
            filename="rows.tsv",
            content="a\tb\tc\n1\t2\t3\n",
            tmp_path=tmp_path,
        )
        assert result.success is True, result.to_dict()
        source = result.updated_state.sources["source"]
        assert source.plugin == "csv"
        assert source.options.get("delimiter") == "\t"

    def test_caller_supplied_delimiter_preserved_on_inline_path(self, tmp_path: Path) -> None:
        result = self._set_pipeline_with_inline_csv_blob(
            filename="rows.tsv",
            content="a;b;c\n1;2;3\n",
            tmp_path=tmp_path,
            extra_source_options={"delimiter": ";"},
        )
        assert result.success is True, result.to_dict()
        source = result.updated_state.sources["source"]
        assert source.options.get("delimiter") == ";"

    def test_csv_inline_blob_does_not_inject_delimiter(self, tmp_path: Path) -> None:
        result = self._set_pipeline_with_inline_csv_blob(
            filename="rows.csv",
            content="a,b,c\n1,2,3\n",
            tmp_path=tmp_path,
        )
        assert result.success is True, result.to_dict()
        source = result.updated_state.sources["source"]
        assert source.options.get("delimiter") is None
