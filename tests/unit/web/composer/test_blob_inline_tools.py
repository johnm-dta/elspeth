"""Composer tool coverage for widened blob_ref inline-content authoring."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from elspeth.contracts.blobs_inline import is_widened_blob_ref
from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.tools import ToolResult, execute_tool, get_tool_definitions
from elspeth.web.composer.yaml_generator import generate_pipeline_dict
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _inline_ref_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={"path": "/tmp/input.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="classify",
                node_type="transform",
                plugin="llm",
                input="rows",
                on_success="classified",
                on_error="discard",
                options={
                    "provider": "openrouter",
                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    "model": "openai/gpt-4o",
                    "prompt_template": "Placeholder",
                    "required_input_fields": [],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(
                name="classified",
                plugin="json",
                options={"path": "/tmp/output.jsonl", "format": "jsonl", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


class _Catalog:
    def list_sources(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
            PluginSummary(name="json", description="JSON source", plugin_type="source", config_fields=[]),
            PluginSummary(name="text", description="Text source", plugin_type="source", config_fields=[]),
        ]

    def list_sinks(self) -> list[PluginSummary]:
        return [PluginSummary(name="json", description="JSON sink", plugin_type="sink", config_fields=[])]

    def list_transforms(self) -> list[PluginSummary]:
        return [PluginSummary(name="llm", description="LLM transform", plugin_type="transform", config_fields=[])]

    def get_schema(self, plugin_type: PluginKind, plugin_name: str) -> PluginSchemaInfo:
        return PluginSchemaInfo(
            name=plugin_name,
            plugin_type=plugin_type,
            description=f"{plugin_name} {plugin_type}",
            json_schema={"title": plugin_name, "properties": {"path": {"type": "string"}}},
            knob_schema={"fields": []},
        )

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        return ()


def _catalog() -> CatalogService:
    return cast(CatalogService, _Catalog())


@pytest.fixture()
def blob_env(tmp_path: Path) -> dict[str, Any]:
    engine = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(engine)
    session_id = "session-inline-blob"
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                title="Inline Blob Test",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            chat_messages_table.insert().values(
                id="user-message-1",
                session_id=session_id,
                role="user",
                content="Use this exact content.",
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
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "blobs").mkdir()
    return {"engine": engine, "session_id": session_id, "data_dir": str(data_dir)}


def _create_blob(
    blob_env: dict[str, Any],
    *,
    filename: str = "prompt.txt",
    mime_type: str = "text/plain",
    content: str = "System prompt",
) -> ToolResult:
    return execute_tool(
        "create_blob",
        {"filename": filename, "mime_type": mime_type, "content": content},
        _empty_state(),
        _catalog(),
        data_dir=blob_env["data_dir"],
        session_engine=blob_env["engine"],
        session_id=blob_env["session_id"],
        user_message_id="user-message-1",
        user_message_content=f"Use this exact content:\n{content}",
    )


def _mark_blob_pending(blob_env: dict[str, Any], blob_id: str) -> None:
    with blob_env["engine"].begin() as conn:
        conn.execute(blobs_table.update().where(blobs_table.c.id == blob_id).values(status="pending"))


class TestListComposerBlobs:
    def test_returns_h4_visibility_shape_without_free_text_or_content(self, blob_env: dict[str, Any]) -> None:
        create = _create_blob(blob_env, filename="prompt.txt", content="Do not leak this prompt.")
        assert create.success is True

        result = execute_tool(
            "list_composer_blobs",
            {},
            _empty_state(),
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is True
        assert list(result.data) == ["blobs"]
        descriptor = result.data["blobs"][0]
        assert set(descriptor) == {"blob_id", "mime_type", "size_bytes", "content_hash", "filename"}
        assert descriptor["blob_id"] == create.data["blob_id"]
        assert descriptor["content_hash"] == create.data["content_hash"]
        assert "source_description" not in descriptor
        assert "content" not in descriptor
        assert "preview" not in descriptor

    def test_only_ready_blobs_are_returned(self, blob_env: dict[str, Any]) -> None:
        ready = _create_blob(blob_env, filename="ready.txt", content="ready")
        pending = _create_blob(blob_env, filename="pending.txt", content="pending")
        _mark_blob_pending(blob_env, pending.data["blob_id"])

        result = execute_tool(
            "list_composer_blobs",
            {},
            _empty_state(),
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is True
        assert [blob["blob_id"] for blob in result.data["blobs"]] == [ready.data["blob_id"]]


class TestWireBlobInlineRef:
    def test_authors_marker_with_authoritative_pinned_hash(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, content="Pinned prompt")
        state = _inline_ref_state()

        result = execute_tool(
            "wire_blob_inline_ref",
            {
                "field_path": "node:classify.options.prompt_template",
                "blob_id": blob.data["blob_id"],
            },
            state,
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is True
        marker = result.updated_state.nodes[0].options["prompt_template"]
        assert marker == {
            "blob_ref": blob.data["blob_id"],
            "mode": "inline_content",
            "sha256": blob.data["content_hash"],
        }

    def test_writes_source_and_output_paths_by_identity(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, content="inline replacement")
        state = _inline_ref_state()

        source_result = execute_tool(
            "wire_blob_inline_ref",
            {
                "field_path": "source.options.schema.description",
                "blob_id": blob.data["blob_id"],
                "encoding": "latin-1",
            },
            state,
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        output_result = execute_tool(
            "wire_blob_inline_ref",
            {
                "field_path": "output:classified.options.header",
                "blob_id": blob.data["blob_id"],
            },
            source_result.updated_state,
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert source_result.success is True
        assert output_result.success is True
        assert "source" in source_result.updated_state.sources
        source_marker = source_result.updated_state.sources["source"].options["schema"]["description"]
        assert source_marker["encoding"] == "latin-1"
        assert output_result.updated_state.outputs[0].options["header"]["mode"] == "inline_content"

    def test_rejects_pending_blob(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, content="pending")
        _mark_blob_pending(blob_env, blob.data["blob_id"])

        result = execute_tool(
            "wire_blob_inline_ref",
            {
                "field_path": "node:classify.options.prompt_template",
                "blob_id": blob.data["blob_id"],
            },
            _inline_ref_state(),
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is False
        assert "not ready" in result.data["error"] or "status" in result.data["error"]

    def test_rejects_llm_typed_disagreeing_hash(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, content="hash source")

        result = execute_tool(
            "wire_blob_inline_ref",
            {
                "field_path": "node:classify.options.prompt_template",
                "blob_id": blob.data["blob_id"],
                "sha256_override": "b" * 64,
            },
            _inline_ref_state(),
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is False
        assert "sha256" in result.data["error"]

    def test_rejects_invalid_field_path(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, content="prompt")

        result = execute_tool(
            "wire_blob_inline_ref",
            {
                "field_path": "transforms[0].options.prompt_template",
                "blob_id": blob.data["blob_id"],
            },
            _inline_ref_state(),
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is False
        assert "field_path" in result.data["error"]

    def test_rejects_unknown_encoding(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, content="prompt")

        result = execute_tool(
            "wire_blob_inline_ref",
            {
                "field_path": "node:classify.options.prompt_template",
                "blob_id": blob.data["blob_id"],
                "encoding": "ascii",
            },
            _inline_ref_state(),
            _catalog(),
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is False
        assert "encoding" in result.data["error"]


class TestSetSourceFromBlobMode:
    def test_set_source_from_blob_emits_explicit_bind_source_mode(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, filename="input.csv", mime_type="text/csv", content="name\nAda")

        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": blob.data["blob_id"], "on_success": "rows", "options": {"schema": {"mode": "observed"}}},
            _empty_state(),
            _catalog(),
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        assert result.success is True
        assert "source" in result.updated_state.sources
        marker = {key: result.updated_state.sources["source"].options[key] for key in ("blob_ref", "mode", "path")}
        assert marker["mode"] == "bind_source"
        shape = is_widened_blob_ref(marker)
        assert shape is not None
        assert shape.mode == "bind_source"
        assert shape.blob_id == UUID(blob.data["blob_id"])

    def test_bind_source_mode_is_stripped_from_engine_yaml(self, blob_env: dict[str, Any]) -> None:
        blob = _create_blob(blob_env, filename="input.csv", mime_type="text/csv", content="name\nAda")
        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": blob.data["blob_id"], "on_success": "rows", "options": {"schema": {"mode": "observed"}}},
            _empty_state(),
            _catalog(),
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )

        pipeline = generate_pipeline_dict(result.updated_state)

        assert "blob_ref" not in pipeline["sources"]["source"]["options"]
        assert "mode" not in pipeline["sources"]["source"]["options"]


def test_tool_definitions_include_inline_blob_authoring_tools() -> None:
    definitions = {definition["name"]: definition for definition in get_tool_definitions()}

    assert "list_composer_blobs" in definitions
    assert definitions["list_composer_blobs"]["parameters"] == {"type": "object", "properties": {}, "required": []}
    assert "wire_blob_inline_ref" in definitions
    assert definitions["wire_blob_inline_ref"]["parameters"]["required"] == ["field_path", "blob_id"]
