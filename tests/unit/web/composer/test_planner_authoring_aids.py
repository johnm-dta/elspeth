"""Live-rendered planner authoring aids: exemplars must validate, byte-for-byte.

The aids exist because the 2026-07-22 pack stress test showed cold planners
fabricating ``blob_id`` values and missing the source options contract: the
pack had rules but no worked exemplars, and the ``no_deployment_plugin_facts``
gate (correctly) forbids plugin literals in the static prompts. These tests
enforce the self-verifying-teaching contract: the exact exemplar objects the
planner prompt carries are run through ``build_set_pipeline_candidate``
against the real catalog, so a drifting exemplar fails CI instead of teaching
planners an invalid shape.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import insert
from sqlalchemy.pool import StaticPool

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.planner_authoring_aids import (
    PLACEHOLDER_BLOB_ID,
    build_planner_authoring_aids,
    source_custody_exemplar_args,
)
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import ToolContext, build_set_pipeline_candidate
from elspeth.web.composer.tools import execute_tool as _dispatch_tool
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _trained_view() -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return PolicyCatalogView.for_trained_operator(catalog, snapshot), snapshot


def _session_with_user_message(content: str) -> tuple[Any, str, str, str]:
    """Session + user chat message whose text contains ``content`` verbatim."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    message_id = str(uuid4())
    message_content = f"Use this exact content:\n{content}"
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="authoring-aids-user",
                auth_provider_type="local",
                title="authoring aids exemplar validation",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=message_id,
                session_id=session_id,
                role="user",
                content=message_content,
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
    return engine, session_id, message_id, message_content


def _custody_context(tmp_path: Path, content: str) -> ToolContext:
    view, snapshot = _trained_view()
    engine, session_id, message_id, message_content = _session_with_user_message(content)
    return ToolContext(
        catalog=view,
        plugin_snapshot=snapshot,
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        user_message_id=message_id,
        user_message_content=message_content,
    )


def _create_real_blob(tmp_path: Path, *, filename: str, mime_type: str, content: str) -> tuple[str, ToolContext]:
    """Create a session blob through the real tool and return (blob_id, context)."""
    view, snapshot = _trained_view()
    engine, session_id, message_id, message_content = _session_with_user_message(content)
    (tmp_path / "blobs").mkdir(exist_ok=True)
    context = ToolContext(
        catalog=view,
        plugin_snapshot=snapshot,
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        user_message_id=message_id,
        user_message_content=message_content,
    )
    result = _dispatch_tool(
        "create_blob",
        {"filename": filename, "mime_type": mime_type, "content": content},
        _empty_state(),
        view,
        plugin_snapshot=snapshot,
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        user_message_id=message_id,
        user_message_content=message_content,
    )
    assert result.success is True, result.to_dict()
    return result.data["blob_id"], context


class TestSourceCustodyExemplar:
    def test_inline_blob_exemplar_validates_through_the_real_candidate_builder(self, tmp_path: Path) -> None:
        """The exact inline-custody exemplar bytes the prompt carries must build."""
        view, _snapshot = _trained_view()
        args = source_custody_exemplar_args(view)
        assert args is not None
        content = args["source"]["inline_blob"]["content"]
        context = _custody_context(tmp_path, content)

        candidate = build_set_pipeline_candidate(args, _empty_state(), context)

        rejection = None if candidate.acceptable else (candidate.result.data or {}).get("error")
        assert candidate.acceptable is True, f"inline custody exemplar rejected: {rejection}"

    def test_existing_blob_exemplar_validates_with_a_real_session_blob(self, tmp_path: Path) -> None:
        """The blob_id binding exemplar must build once given a tool-returned id."""
        view, _snapshot = _trained_view()
        placeholder_args = source_custody_exemplar_args(view, blob_id=PLACEHOLDER_BLOB_ID)
        assert placeholder_args is not None
        inline_args = source_custody_exemplar_args(view)
        assert inline_args is not None
        blob_id, context = _create_real_blob(
            tmp_path,
            filename=inline_args["source"]["inline_blob"]["filename"],
            mime_type=inline_args["source"]["inline_blob"]["mime_type"],
            content=inline_args["source"]["inline_blob"]["content"],
        )
        args = source_custody_exemplar_args(view, blob_id=blob_id)
        assert args is not None

        candidate = build_set_pipeline_candidate(args, _empty_state(), context)

        rejection = None if candidate.acceptable else (candidate.result.data or {}).get("error")
        assert candidate.acceptable is True, f"existing-blob exemplar rejected: {rejection}"
        # Single-source contract: only the binding differs between the two
        # variants the prompt shows; everything downstream is byte-identical.
        assert {key: value for key, value in args.items() if key != "source"} == {
            key: value for key, value in placeholder_args.items() if key != "source"
        }
        assert placeholder_args["source"]["blob_id"] == PLACEHOLDER_BLOB_ID

    def test_exemplar_variants_differ_only_in_the_custody_binding(self) -> None:
        view, _snapshot = _trained_view()
        inline_args = source_custody_exemplar_args(view)
        blob_args = source_custody_exemplar_args(view, blob_id=PLACEHOLDER_BLOB_ID)
        assert inline_args is not None and blob_args is not None

        assert "inline_blob" in inline_args["source"]
        assert "blob_id" not in inline_args["source"]
        assert "blob_id" in blob_args["source"]
        assert "inline_blob" not in blob_args["source"]
        # Neither variant authors custody-owned fields by hand.
        for variant in (inline_args, blob_args):
            assert "path" not in variant["source"]["options"]
            assert "blob_ref" not in variant["source"]["options"]
            assert variant["source"]["options"]["schema"]["mode"]
            assert variant["source"]["on_validation_failure"]


class TestAuthoringAidsPayload:
    def test_payload_carries_the_custody_exemplar_and_the_closed_provenance_rule(self) -> None:
        view, _snapshot = _trained_view()

        payload = build_planner_authoring_aids(view)

        custody = payload["source_custody"]
        assert custody["set_pipeline_exemplar_inline_blob"] == source_custody_exemplar_args(view)
        blob_variant = source_custody_exemplar_args(view, blob_id=PLACEHOLDER_BLOB_ID)
        assert blob_variant is not None
        assert custody["existing_blob_source_binding"] == blob_variant["source"]
        rules = " ".join(custody["rules"])
        assert "ONLY from blob-tool output" in rules
        assert "inline_blob" in rules
        assert "Never fabricate" in rules

    def test_payload_never_carries_a_fabricated_blob_identifier(self) -> None:
        """The prompt teaches provenance; it must not model a fake UUID itself."""
        view, _snapshot = _trained_view()

        rendered = json.dumps(build_planner_authoring_aids(view))

        assert PLACEHOLDER_BLOB_ID in rendered
        assert "-4000-" not in rendered  # no synthetic UUID4-shaped identifiers
        assert rendered.count("blob_id") >= 1

    def test_payload_is_canonical_json_compatible(self) -> None:
        from elspeth.core.canonical import canonical_json

        view, _snapshot = _trained_view()

        canonical_json(build_planner_authoring_aids(view))
