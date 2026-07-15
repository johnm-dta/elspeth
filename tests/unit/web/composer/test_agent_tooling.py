"""Tests for B1-B5 Agent Tooling features (Composer MCP).

B1: create_blob, update_blob, delete_blob, get_blob_content
B2: Structured validation payloads from all mutation tools
B3: Source patching parity (verified by existing tests — this adds edge cases)
B4: Blob ID abstraction over raw storage paths
B5: Pipeline diff/change summary tool
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import func, insert, select

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import redact_source_storage_path
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.tools import (
    _ALLOWED_BLOB_MIME_TYPES,
    ToolResult,
    diff_states,
)
from elspeth.web.composer.tools import (
    execute_tool as _execute_tool,
)
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, chat_messages_table
from elspeth.web.sessions.schema import initialize_session_schema

EXPECTED_REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"


class _CatalogDouble:
    def list_sources(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
            PluginSummary(name="json", description="JSON source", plugin_type="source", config_fields=[]),
            PluginSummary(name="text", description="Text source", plugin_type="source", config_fields=[]),
        ]

    def list_transforms(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="passthrough", description="Passthrough transform", plugin_type="transform", config_fields=[]),
        ]

    def list_sinks(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="csv", description="CSV sink", plugin_type="sink", config_fields=[]),
        ]

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        return PluginSchemaInfo(
            name=name,
            plugin_type=plugin_type,
            description=f"{name} {plugin_type}",
            json_schema={"title": f"{name.title()}Config", "properties": {"path": {"type": "string"}}},
            knob_schema={"fields": []},
        )

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: Any,
    ) -> tuple[str, ...]:
        return ()


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _mock_catalog() -> CatalogService:
    return _CatalogDouble()


def _insert_user_message(engine: Any, session_id: str, content: str) -> str:
    """Seed the route-level user message that verbatim blob provenance binds to."""

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


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: Any,
    data_dir: str | None = None,
    session_engine: Any | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    supplied_snapshot = kwargs.pop("plugin_snapshot", None)
    if isinstance(catalog, PolicyCatalogView):
        if not isinstance(supplied_snapshot, PluginAvailabilitySnapshot):
            raise AssertionError("policy catalog tests must supply their exact snapshot")
        policy_catalog = catalog
        snapshot = supplied_snapshot
    else:
        if supplied_snapshot is not None:
            raise AssertionError("a snapshot requires its matching policy catalog")
        snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
        policy_catalog = PolicyCatalogView.for_trained_operator(catalog, snapshot)
    if (
        tool_name in {"create_blob", "update_blob"}
        and session_engine is not None
        and session_id is not None
        and "user_message_id" not in kwargs
        and "user_message_content" not in kwargs
    ):
        content = arguments.get("content")
        if isinstance(content, str):
            user_message_content = f"Use this exact content:\n{content}"
            kwargs["user_message_id"] = _insert_user_message(session_engine, session_id, user_message_content)
            kwargs["user_message_content"] = user_message_content
    return _execute_tool(
        tool_name,
        arguments,
        state,
        policy_catalog,
        plugin_snapshot=snapshot,
        data_dir=data_dir,
        session_engine=session_engine,
        session_id=session_id,
        **kwargs,
    )


@pytest.fixture()
def blob_env(tmp_path: Path) -> dict[str, Any]:
    """Create a temporary session database and data directory for blob tests.

    Inserts a real ``sessions`` row so FK-enforced blob inserts succeed.
    Previously these tests relied on SQLite silently skipping FK
    enforcement; the engine factory now enables PRAGMA foreign_keys=ON,
    so the session row must actually exist.
    """
    from datetime import UTC, datetime

    from elspeth.web.sessions.models import sessions_table

    engine = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(engine)

    session_id = "test-session-001"
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

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "blobs").mkdir()
    return {
        "engine": engine,
        "data_dir": str(data_dir),
        "session_id": session_id,
    }


# ── B1: Blob CRUD ──────────────────────────────────────────────────────


class TestCreateBlob:
    def test_creates_blob_with_content(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "create_blob",
            {"filename": "urls.csv", "mime_type": "text/csv", "content": "url\nhttps://example.com"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is True
        assert result.data["filename"] == "urls.csv"
        assert result.data["mime_type"] == "text/csv"
        assert result.data["size_bytes"] == len(b"url\nhttps://example.com")
        assert result.data["blob_id"]
        assert result.data["content_hash"]

    def test_rejects_unsupported_mime_type(self, blob_env: dict[str, Any]) -> None:
        # _prepare_blob_create raises ToolArgumentError (CEC1 channel
        # discipline) for MIME types outside the operator allowlist —
        # propagates to the compose loop's ARG_ERROR branch, not
        # masked as SUCCESS-with-success=False. Leak-prevention: the
        # rejected mime_type value is NOT echoed in the message;
        # only the allowlist appears.
        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "create_blob",
                {"filename": "image.png", "mime_type": "image/png", "content": "fake"},
                state,
                catalog,
                data_dir=blob_env["data_dir"],
                session_engine=blob_env["engine"],
                session_id=blob_env["session_id"],
            )
        assert exc_info.value.argument == "mime_type"
        assert "one of:" in exc_info.value.expected
        # Leak-prevention pin: the rejected value must not appear in args[0].
        assert "image/png" not in exc_info.value.args[0]

    def test_requires_session_context(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "create_blob",
            {"filename": "x.csv", "mime_type": "text/csv", "content": "a"},
            state,
            catalog,
        )
        assert result.success is False
        assert "session context" in result.data["error"]

    def test_writes_file_to_disk(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        content = '{"key": "value"}'
        result = execute_tool(
            "create_blob",
            {"filename": "data.json", "mime_type": "application/json", "content": content},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = result.data["blob_id"]
        # Verify file exists on disk under session subdirectory
        session_dir = Path(blob_env["data_dir"]) / "blobs" / blob_env["session_id"]
        files = list(session_dir.iterdir())
        assert len(files) == 1
        assert blob_id in files[0].name
        assert files[0].read_text() == content


class TestUpdateBlob:
    def test_updates_content(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # Create
        create_result = execute_tool(
            "create_blob",
            {"filename": "data.csv", "mime_type": "text/csv", "content": "old"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        # Update
        update_result = execute_tool(
            "update_blob",
            {"blob_id": blob_id, "content": "new content"},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert update_result.success is True
        assert update_result.data["size_bytes"] == len(b"new content")
        assert update_result.data["content_hash"] != create_result.data["content_hash"]

    def test_not_found(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": "nonexistent", "content": "x"},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is False
        # update_blob validates blob_id as a canonical UUID at the Tier-3
        # boundary before the row lookup, so a non-UUID literal surfaces the
        # more specific "not a valid UUID" repair hint rather than "not found".
        assert "is not a valid UUID" in result.data["error"]


class TestDeleteBlob:
    def test_deletes_blob_and_file(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # Create
        create_result = execute_tool(
            "create_blob",
            {"filename": "temp.txt", "mime_type": "text/plain", "content": "hello"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        # Delete
        delete_result = execute_tool(
            "delete_blob",
            {"blob_id": blob_id},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert delete_result.success is True
        assert delete_result.data["deleted"] is True
        # Verify file is gone from session subdirectory
        session_dir = Path(blob_env["data_dir"]) / "blobs" / blob_env["session_id"]
        assert list(session_dir.iterdir()) == []
        # Verify DB record is gone
        with blob_env["engine"].connect() as conn:
            rows = conn.execute(select(blobs_table)).fetchall()
            assert len(rows) == 0


class TestGetBlobContent:
    def test_retrieves_content(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        content = "line1\nline2\nline3"
        create_result = execute_tool(
            "create_blob",
            {"filename": "lines.txt", "mime_type": "text/plain", "content": content},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        get_result = execute_tool(
            "get_blob_content",
            {"blob_id": blob_id},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert get_result.success is True
        assert get_result.data["content"] == content
        assert get_result.data["truncated"] is False

    def test_not_found(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_blob_content",
            {"blob_id": "ghost"},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is False


# ── B2: Structured Validation Payloads ──────────────────────────────────


class TestStructuredValidation:
    def test_errors_have_component_and_severity(self) -> None:
        state = _empty_state()
        v = state.validate()
        assert len(v.errors) >= 2
        for entry in v.errors:
            assert entry.component
            assert entry.message
            assert entry.severity in ("high", "medium", "low")

    def test_warnings_have_component_and_severity(self) -> None:
        source = SourceSpec(plugin="csv", on_success="dangling", options={}, on_validation_failure="quarantine")
        output = OutputSpec(name="main", plugin="csv", options={}, on_write_failure="discard")
        state = _empty_state().with_source(source).with_output(output)
        v = state.validate()
        # Source on_success='dangling' doesn't match any node input
        assert any(e.component == "source" for e in v.warnings)

    def test_suggestions_have_component(self) -> None:
        source = SourceSpec(plugin="csv", on_success="main", options={}, on_validation_failure="quarantine")
        output = OutputSpec(name="main", plugin="csv", options={}, on_write_failure="discard")
        state = _empty_state().with_source(source).with_output(output)
        v = state.validate()
        assert any(e.component == "source" for e in v.suggestions)

    def test_tool_result_serializes_structured_entries(self) -> None:
        state = _empty_state()
        v = state.validate()
        result = ToolResult(success=True, updated_state=state, validation=v, affected_nodes=())
        d = result.to_dict()
        for entry in d["validation"]["errors"]:
            assert "component" in entry
            assert "message" in entry
            assert "severity" in entry

    def test_all_mutation_tools_return_validation(self) -> None:
        """Every mutation tool returns is_valid + errors + warnings + suggestions."""
        catalog = _mock_catalog()
        state = _empty_state()
        result = execute_tool(
            "set_source",
            {"plugin": "csv", "on_success": "t1", "options": {"schema": {"mode": "observed"}}, "on_validation_failure": "quarantine"},
            state,
            catalog,
        )
        d = result.to_dict()
        v = d["validation"]
        assert "is_valid" in v
        assert "errors" in v
        assert "warnings" in v
        assert "suggestions" in v


# ── B4: Blob ID Abstraction ─────────────────────────────────────────────


class TestRedactStoragePaths:
    def test_redacts_path_when_blob_ref_present(self) -> None:
        state_dict = {
            "source": {
                "plugin": "csv",
                "options": {"path": "/internal/blobs/abc123_data.csv", "blob_ref": "abc123"},
                "on_success": "t1",
            },
            "nodes": [],
            "edges": [],
            "outputs": [],
        }
        redacted = redact_source_storage_path(state_dict)
        assert redacted["source"]["options"]["path"] == EXPECTED_REDACTED_BLOB_SOURCE_PATH
        assert redacted["source"]["options"]["blob_ref"] == "abc123"
        assert "/internal/blobs/abc123_data.csv" not in str(redacted)

    def test_preserves_path_without_blob_ref(self) -> None:
        state_dict = {
            "source": {
                "plugin": "csv",
                "options": {"path": "/data/blobs/manual.csv"},
                "on_success": "t1",
            },
            "nodes": [],
        }
        redacted = redact_source_storage_path(state_dict)
        assert redacted["source"]["options"]["path"] == "/data/blobs/manual.csv"

    def test_no_source_passthrough(self) -> None:
        state_dict: dict[str, Any] = {"source": None, "nodes": []}
        assert redact_source_storage_path(state_dict) is state_dict

    def test_does_not_mutate_original(self) -> None:
        state_dict = {
            "source": {
                "options": {"path": "/x", "blob_ref": "b1"},
            },
        }
        redacted = redact_source_storage_path(state_dict)
        # Original unchanged
        assert "path" in state_dict["source"]["options"]
        # Redacted version different
        assert redacted["source"]["options"]["path"] == EXPECTED_REDACTED_BLOB_SOURCE_PATH


# ── B5: Pipeline Diff ───────────────────────────────────────────────────


class TestDiffStates:
    def test_empty_to_empty_no_changes(self) -> None:
        s = _empty_state()
        diff = diff_states(s, s)
        assert diff["total_changes"] == 0
        assert diff["sources_changed"] is False

    def test_source_added(self) -> None:
        s1 = _empty_state()
        source = SourceSpec(plugin="csv", on_success="t1", options={}, on_validation_failure="quarantine")
        s2 = s1.with_source(source)
        diff = diff_states(s1, s2)
        assert diff["sources_changed"] is True
        assert diff["sources"]["added"] == ["source"]

    def test_source_removed(self) -> None:
        source = SourceSpec(plugin="csv", on_success="t1", options={}, on_validation_failure="quarantine")
        s1 = _empty_state().with_source(source)
        s2 = s1.without_source()
        diff = diff_states(s1, s2)
        assert diff["sources_changed"] is True
        assert diff["sources"]["removed"] == ["source"]

    def test_node_added(self) -> None:
        s1 = _empty_state()
        node = NodeSpec(
            id="n1",
            node_type="transform",
            plugin="passthrough",
            input="in",
            on_success="out",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        s2 = s1.with_node(node)
        diff = diff_states(s1, s2)
        assert "n1" in diff["nodes"]["added"]
        assert diff["total_changes"] >= 1

    def test_node_modified(self) -> None:
        node1 = NodeSpec(
            id="n1",
            node_type="transform",
            plugin="passthrough",
            input="in",
            on_success="out",
            on_error=None,
            options={"x": 1},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        node2 = NodeSpec(
            id="n1",
            node_type="transform",
            plugin="passthrough",
            input="in",
            on_success="out",
            on_error=None,
            options={"x": 2},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        s1 = _empty_state().with_node(node1)
        s2 = s1.with_node(node2)
        diff = diff_states(s1, s2)
        assert "n1" in diff["nodes"]["modified"]

    def test_node_removed(self) -> None:
        node = NodeSpec(
            id="n1",
            node_type="transform",
            plugin="passthrough",
            input="in",
            on_success="out",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        s1 = _empty_state().with_node(node)
        s2 = s1.without_node("n1")
        assert s2 is not None
        diff = diff_states(s1, s2)
        assert "n1" in diff["nodes"]["removed"]

    def test_output_added_and_removed(self) -> None:
        output = OutputSpec(name="main", plugin="csv", options={}, on_write_failure="discard")
        s1 = _empty_state()
        s2 = s1.with_output(output)
        diff = diff_states(s1, s2)
        assert "main" in diff["outputs"]["added"]

        diff_reverse = diff_states(s2, s1)
        assert "main" in diff_reverse["outputs"]["removed"]

    def test_warnings_introduced_and_resolved(self) -> None:
        source = SourceSpec(plugin="csv", on_success="dangling", options={}, on_validation_failure="quarantine")
        output = OutputSpec(name="main", plugin="csv", options={}, on_write_failure="discard")
        s1 = _empty_state().with_output(output)
        s2 = s1.with_source(source)
        diff = diff_states(s1, s2)
        # s2 introduces W2: source on_success points to non-existent target
        assert any("on_success 'dangling'" in w and "data may not flow" in w for w in diff["warnings_introduced"]), (
            f"Expected dangling on_success warning, got: {diff['warnings_introduced']}"
        )

    def test_version_tracking(self) -> None:
        s1 = _empty_state()
        s2 = s1.with_metadata({"name": "Updated"})
        diff = diff_states(s1, s2)
        assert diff["from_version"] == 1
        assert diff["to_version"] == 2


class TestDiffPipelineTool:
    def test_returns_error_without_baseline(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("diff_pipeline", {}, state, catalog)
        assert result.success is True
        assert "No baseline" in result.data["error"]

    def test_returns_changes_with_baseline(self) -> None:
        s1 = _empty_state()
        source = SourceSpec(plugin="csv", on_success="t1", options={}, on_validation_failure="quarantine")
        s2 = s1.with_source(source)
        catalog = _mock_catalog()
        result = execute_tool("diff_pipeline", {}, s2, catalog, baseline=s1)
        assert result.success is True
        assert result.data["sources_changed"] is True
        assert result.data["total_changes"] >= 1


# ── B1/B2 Integration: Allowed MIME types ───────────────────────────────


class TestAllowedMimeTypes:
    def test_all_mime_types_in_tool_definition(self) -> None:
        from elspeth.web.composer.tools import get_tool_definitions

        defs = get_tool_definitions()
        create_blob_def = next(d for d in defs if d["name"] == "create_blob")
        enum_types = set(create_blob_def["parameters"]["properties"]["mime_type"]["enum"])
        assert enum_types == _ALLOWED_BLOB_MIME_TYPES


# ── QA Review: Missing Tests ────────────────────────────────────────────


class TestCreateBlobSecurity:
    """Path traversal and filename sanitization."""

    def test_path_traversal_filename_is_sanitized(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "create_blob",
            {"filename": "../../etc/evil.csv", "mime_type": "text/csv", "content": "x"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is True
        # Sanitized filename should be just the basename
        assert result.data["filename"] == "evil.csv"

    def test_dot_dot_filename_rejected(self, blob_env: dict[str, Any]) -> None:
        # sanitize_filename ValueError is wrapped by ToolArgumentError
        # (CEC1 channel discipline). The original cause carrying the
        # offending filename remains on __cause__ for auditors but
        # is NOT echoed to the LLM via args[0].
        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "create_blob",
                {"filename": "..", "mime_type": "text/csv", "content": "x"},
                state,
                catalog,
                data_dir=blob_env["data_dir"],
                session_engine=blob_env["engine"],
                session_id=blob_env["session_id"],
            )
        assert exc_info.value.argument == "filename"
        # The original sanitize_filename ValueError is preserved on __cause__.
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_content_hash_is_valid_sha256(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "create_blob",
            {"filename": "test.txt", "mime_type": "text/plain", "content": "hello"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert len(result.data["content_hash"]) == 64

    def test_blob_id_is_valid_uuid(self, blob_env: dict[str, Any]) -> None:
        """UUID format matches BlobServiceImpl (str(uuid4())), not hex[:24]."""
        from uuid import UUID

        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "create_blob",
            {"filename": "test.txt", "mime_type": "text/plain", "content": "x"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        # Should be parseable as a standard UUID (36-char with hyphens)
        UUID(result.data["blob_id"])


class TestGetBlobContentTruncation:
    def test_truncates_large_content(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        large_content = "x" * 60_000
        create_result = execute_tool(
            "create_blob",
            {"filename": "big.txt", "mime_type": "text/plain", "content": large_content},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        get_result = execute_tool(
            "get_blob_content",
            {"blob_id": create_result.data["blob_id"]},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert get_result.success is True
        assert get_result.data["truncated"] is True
        assert len(get_result.data["content"]) == 50_000


class TestUpdateBlobFileOnDisk:
    def test_file_content_actually_changes(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        create_result = execute_tool(
            "create_blob",
            {"filename": "data.csv", "mime_type": "text/csv", "content": "old"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        execute_tool(
            "update_blob",
            {"blob_id": blob_id, "content": "new content"},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        # Verify file on disk was actually overwritten
        with blob_env["engine"].connect() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).first()
            assert Path(row.storage_path).read_text() == "new content"


class TestDeleteBlobMissingFile:
    def test_succeeds_when_file_already_missing(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        create_result = execute_tool(
            "create_blob",
            {"filename": "temp.txt", "mime_type": "text/plain", "content": "x"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        # Manually delete the file before calling delete_blob
        with blob_env["engine"].connect() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).first()
            Path(row.storage_path).unlink()
        # delete_blob should still succeed (DB record cleanup)
        result = execute_tool(
            "delete_blob",
            {"blob_id": blob_id},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is True


class TestCrossSessionIsolation:
    def test_blob_not_accessible_from_other_session(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        create_result = execute_tool(
            "create_blob",
            {"filename": "secret.csv", "mime_type": "text/csv", "content": "data"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        # Try to access from a different session
        result = execute_tool(
            "get_blob_content",
            {"blob_id": blob_id},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id="other-session-999",
        )
        assert result.success is False
        assert "not found" in result.data["error"]

    def test_delete_scoped_to_session(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        create_result = execute_tool(
            "create_blob",
            {"filename": "mine.csv", "mime_type": "text/csv", "content": "data"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        # Try to delete from a different session — should fail (not found)
        result = execute_tool(
            "delete_blob",
            {"blob_id": blob_id},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id="other-session-999",
        )
        assert result.success is False


class TestDiffStatesEdges:
    def test_edge_added(self) -> None:
        from elspeth.web.composer.state import EdgeSpec

        s1 = _empty_state()
        edge = EdgeSpec(id="e1", from_node="source", to_node="n1", edge_type="on_success", label=None)
        s2 = s1.with_edge(edge)
        diff = diff_states(s1, s2)
        assert "e1" in diff["edges"]["added"]

    def test_edge_removed(self) -> None:
        from elspeth.web.composer.state import EdgeSpec

        edge = EdgeSpec(id="e1", from_node="source", to_node="n1", edge_type="on_success", label=None)
        s1 = _empty_state().with_edge(edge)
        s2 = s1.without_edge("e1")
        assert s2 is not None
        diff = diff_states(s1, s2)
        assert "e1" in diff["edges"]["removed"]

    def test_edge_modified(self) -> None:
        from elspeth.web.composer.state import EdgeSpec

        e1 = EdgeSpec(id="e1", from_node="source", to_node="n1", edge_type="on_success", label=None)
        e2 = EdgeSpec(id="e1", from_node="source", to_node="n2", edge_type="on_success", label=None)
        s1 = _empty_state().with_edge(e1)
        s2 = s1.with_edge(e2)
        diff = diff_states(s1, s2)
        assert "e1" in diff["edges"]["modified"]


class TestDiffStatesWarningsResolved:
    def test_warning_resolved(self) -> None:
        source = SourceSpec(plugin="csv", on_success="dangling", options={}, on_validation_failure="quarantine")
        output = OutputSpec(name="main", plugin="csv", options={}, on_write_failure="discard")
        # s1 has source pointing to "dangling" (generates warning)
        s1 = _empty_state().with_source(source).with_output(output)
        # s2 fixes the source to point to "main"
        fixed_source = SourceSpec(plugin="csv", on_success="main", options={}, on_validation_failure="quarantine")
        s2 = s1.with_source(fixed_source)
        diff = diff_states(s1, s2)
        # Fixing on_success resolves W2 (dangling target) and W1 (unreferenced output)
        assert any("on_success 'dangling'" in w and "data may not flow" in w for w in diff["warnings_resolved"]), (
            f"Expected dangling on_success resolved, got: {diff['warnings_resolved']}"
        )
        assert any("Output 'main'" in w and "never receive data" in w for w in diff["warnings_resolved"]), (
            f"Expected unreferenced output resolved, got: {diff['warnings_resolved']}"
        )

    def test_metadata_changed_tracked(self) -> None:
        s1 = _empty_state()
        s2 = s1.with_metadata({"name": "Updated"})
        diff = diff_states(s1, s2)
        assert diff["metadata_changed"] is True
        assert diff["total_changes"] >= 1

    def test_precomputed_validation_matches_fresh(self) -> None:
        """diff_states with pre-computed validations produces same result as fresh."""
        source = SourceSpec(plugin="csv", on_success="dangling", options={}, on_validation_failure="quarantine")
        output = OutputSpec(name="main", plugin="csv", options={}, on_write_failure="discard")
        s1 = _empty_state().with_output(output)
        s2 = s1.with_source(source)

        # Fresh (no pre-computed)
        diff_fresh = diff_states(s1, s2)

        # Pre-computed
        diff_threaded = diff_states(
            s1,
            s2,
            baseline_validation=s1.validate(),
            current_validation=s2.validate(),
        )

        assert diff_fresh["warnings_introduced"] == diff_threaded["warnings_introduced"]
        assert diff_fresh["warnings_resolved"] == diff_threaded["warnings_resolved"]
        assert diff_fresh["total_changes"] == diff_threaded["total_changes"]


class TestSetSourceFromBlob:
    def test_wires_blob_as_source(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        create_result = execute_tool(
            "create_blob",
            {"filename": "data.csv", "mime_type": "text/csv", "content": "a,b\n1,2"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        blob_id = create_result.data["blob_id"]
        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": blob_id, "on_success": "step1", "options": {"schema": {"mode": "observed"}}},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is True
        source = result.updated_state.sources["source"]
        assert source.plugin == "csv"
        assert source.options["blob_ref"] == blob_id

    def test_blob_not_found(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": "nonexistent", "on_success": "step1"},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is False

    def test_unsupported_mime_type(self, blob_env: dict[str, Any]) -> None:
        """A manually corrupted blob row trips the Tier 1 MIME guard."""
        from datetime import UTC, datetime

        state = _empty_state()
        catalog = _mock_catalog()
        # blob_id must be a canonical UUID: set_source_from_blob now
        # rejects malformed ids at the Tier-3 argument boundary, so a
        # non-UUID literal would short-circuit before the row is read and
        # the Tier-1 MIME guard under test would never run.
        exotic_blob_id = str(uuid4())
        with blob_env["engine"].begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=exotic_blob_id,
                    session_id=blob_env["session_id"],
                    filename="data.parquet",
                    mime_type="application/x-parquet",
                    size_bytes=100,
                    # ck_blobs_ready_hash: ready rows must carry a
                    # SHA-256 hex digest.  The exact value is immaterial
                    # for this test (we only check MIME-inference error
                    # handling) — any well-formed 64-char lowercase hex
                    # satisfies the constraint.
                    content_hash="0" * 64,
                    storage_path="/tmp/fake",
                    created_at=datetime.now(UTC),
                    created_by="user",
                    status="ready",
                )
            )
        with pytest.raises(AuditIntegrityError, match=r"blobs\.mime_type is 'application/x-parquet'"):
            execute_tool(
                "set_source_from_blob",
                {"blob_id": exotic_blob_id, "on_success": "step1"},
                state,
                catalog,
                session_engine=blob_env["engine"],
                session_id=blob_env["session_id"],
            )


class TestCreateBlobToSetSourceEndToEnd:
    """E2E: create_blob then set_source_from_blob wires a valid CSV source."""

    def test_create_then_set_source_produces_valid_state(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        # Step 1: Create a blob with inline CSV content
        csv_content = "name,age\nAlice,30\nBob,25"
        create_result = execute_tool(
            "create_blob",
            {"filename": "people.csv", "mime_type": "text/csv", "content": csv_content},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert create_result.success is True
        blob_id = create_result.data["blob_id"]

        # Step 2: Wire the blob as the pipeline source
        source_result = execute_tool(
            "set_source_from_blob",
            {"blob_id": blob_id, "on_success": "transform1", "options": {"schema": {"mode": "observed"}}},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert source_result.success is True

        # Step 3: Verify the resulting CompositionState
        new_state = source_result.updated_state
        source = new_state.sources["source"]
        assert source.plugin == "csv"
        assert source.options["blob_ref"] == blob_id
        assert source.on_success == "transform1"

        # The storage path should point to the actual file on disk
        storage_path = Path(source.options["path"])
        assert storage_path.exists()
        assert storage_path.read_text() == csv_content


class TestListBlobsAndMetadata:
    def test_list_blobs_empty(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "list_blobs",
            {},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is True
        assert len(result.data) == 0

    def test_list_blobs_returns_entries(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        execute_tool(
            "create_blob",
            {"filename": "a.csv", "mime_type": "text/csv", "content": "x"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        result = execute_tool(
            "list_blobs",
            {},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert len(result.data) == 1
        # storage_path should NOT be in list response
        assert "storage_path" not in result.data[0]

    def test_get_blob_metadata_excludes_storage_path(self, blob_env: dict[str, Any]) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        create_result = execute_tool(
            "create_blob",
            {"filename": "a.csv", "mime_type": "text/csv", "content": "x"},
            state,
            catalog,
            data_dir=blob_env["data_dir"],
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        result = execute_tool(
            "get_blob_metadata",
            {"blob_id": create_result.data["blob_id"]},
            state,
            catalog,
            session_engine=blob_env["engine"],
            session_id=blob_env["session_id"],
        )
        assert result.success is True
        assert "storage_path" not in result.data
