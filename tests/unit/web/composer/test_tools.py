"""Tests for composition tools — discovery delegation and mutation + validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import event, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginSchemaInfo,
    PluginSecretRequirement,
    PluginSummary,
)
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationEntry,
    ValidationSummary,
)
from elspeth.web.composer.tools import (
    ToolResult,
    _apply_merge_patch,
    _compute_validation_delta,
    _credential_wiring_contract_failure,
    _failure_result,
    _inject_prior_validation,
    _prevalidate_plugin_options,
    execute_tool,
    get_expression_grammar,
    get_tool_definitions,
)
from elspeth.web.execution.schemas import ValidationCheck, ValidationError, ValidationReadiness, ValidationResult
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    PROMPT_SHIELD_USER_TERM,
    PROMPT_TEMPLATE_PARTS_KEY,
    SOURCE_AUTHORING_KEY,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema

# Stub SHA-256 hex digest for test fixtures.  Must satisfy the
# ``ck_blobs_ready_hash`` invariant — exactly 64 lowercase hex
# characters — even when the surrounding test does not actually verify
# the hash.  Using a structurally valid placeholder keeps the fixtures
# from accidentally exercising the malformed-hash bypass path the
# database CHECK was added to close.
_STUB_SHA256 = "a" * 64
_STUB_SHA256_ALT = "b" * 64
EXPECTED_REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _default_source(state: CompositionState) -> SourceSpec | None:
    return state.sources.get("source")


def _pipeline_state_default_source(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    sources = data["sources"]
    if not isinstance(sources, Mapping):
        raise AssertionError("get_pipeline_state data['sources'] must be a mapping")
    source = sources.get("source")
    if source is not None and not isinstance(source, Mapping):
        raise AssertionError("get_pipeline_state data['sources']['source'] must be a mapping")
    return source


def _mock_catalog() -> MagicMock:
    """Mock CatalogService with real PluginSummary/PluginSchemaInfo instances.

    AC #16: Tests must use real PluginSummary and PluginSchemaInfo instances,
    not plain dicts. Mock return types must match the CatalogService protocol.
    """
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv",
            description="CSV file source",
            plugin_type="source",
            config_fields=[
                ConfigFieldSummary(name="path", type="string", required=True, description="File path", default=None),
            ],
        ),
        PluginSummary(
            name="text",
            description="Text line source",
            plugin_type="source",
            config_fields=[],
        ),
        PluginSummary(
            name="json",
            description="JSON file source",
            plugin_type="source",
            config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(
            name="passthrough",
            description="Uppercase transform",
            plugin_type="transform",
            config_fields=[],
        ),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(
            name="csv",
            description="CSV file sink",
            plugin_type="sink",
            config_fields=[],
        ),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV file source",
        json_schema={"title": "CsvSourceConfig", "properties": {"path": {"type": "string"}}},
        knob_schema={"fields": []},
    )
    return catalog


def _session_engine_with_session() -> tuple[Any, str]:
    """Create a session DB with one session row for blob-tool tests."""
    from datetime import UTC, datetime

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
    """Persist a user chat message for verbatim blob provenance tests."""
    from datetime import UTC, datetime

    user_message_id = str(uuid4())
    with engine.begin() as conn:
        latest = conn.execute(
            select(chat_messages_table.c.sequence_no)
            .where(chat_messages_table.c.session_id == session_id)
            .order_by(chat_messages_table.c.sequence_no.desc())
        ).first()
        sequence_no = 1 if latest is None else latest.sequence_no + 1
        conn.execute(
            chat_messages_table.insert().values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content=content,
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=sequence_no,
                writer_principal="route_user_message",
                created_at=datetime.now(UTC),
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return user_message_id


def _verbatim_blob_context(engine: Any, session_id: str, content: str) -> dict[str, str]:
    """Return execute_tool kwargs proving ``content`` came from a user message."""
    user_message_content = f"Use this exact content:\n{content}"
    user_message_id = _insert_user_message(engine, session_id, user_message_content)
    return {
        "user_message_id": user_message_id,
        "user_message_content": user_message_content,
    }


def test_execute_create_blob_honors_configured_session_quota(tmp_path: Path) -> None:
    engine, session_id = _session_engine_with_session()
    result = execute_tool(
        "create_blob",
        {"filename": "too-large.txt", "mime_type": "text/plain", "content": "exceeds"},
        _empty_state(),
        _mock_catalog(),
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        max_blob_storage_per_session_bytes=3,
        **_verbatim_blob_context(engine, session_id, "exceeds"),
    )

    assert result.success is False
    assert "3 byte limit" in result.data["error"]
    assert list((tmp_path / "blobs" / session_id).glob("*")) == []


class TestToolResult:
    def test_frozen(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=False, errors=(ValidationEntry("test", "err", "high"),)),
            affected_nodes=("n1", "n2"),
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_affected_nodes_frozen(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=("n1",),
        )
        assert isinstance(result.affected_nodes, tuple)

    def test_to_dict_includes_warnings_and_suggestions(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(
                is_valid=True,
                errors=(),
                warnings=(ValidationEntry("test", "warn1", "medium"), ValidationEntry("test", "warn2", "medium")),
                suggestions=(ValidationEntry("test", "sug1", "low"),),
            ),
            affected_nodes=(),
        )
        d = result.to_dict()
        assert d["validation"]["warnings"] == [
            {"component": "test", "message": "warn1", "severity": "medium"},
            {"component": "test", "message": "warn2", "severity": "medium"},
        ]
        assert d["validation"]["suggestions"] == [
            {"component": "test", "message": "sug1", "severity": "low"},
        ]

    def test_to_dict_empty_warnings_and_suggestions(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=(),
        )
        d = result.to_dict()
        assert d["validation"]["warnings"] == []
        assert d["validation"]["suggestions"] == []


class TestFailureResult:
    """``_failure_result`` must lead validation.errors with the rejection reason.

    The composer LLM converges via ``validation.errors`` ordering — see
    composer session 58d7ede3 (2026-05-08) where the LLM read stale
    state-snapshot errors first and burned a full round retrying the
    same call shape with only a cosmetic change. Locking the leading
    entry here makes that regression invisible to refactors.
    """

    def test_prepends_rejection_reason_to_errors(self) -> None:
        state = _empty_state()  # has neither source nor sinks
        result = _failure_result(state, "boom: missing path")

        assert result.success is False
        assert result.validation.is_valid is False
        first = result.validation.errors[0]
        assert first.component == "rejected_mutation"
        assert first.message == "boom: missing path"
        assert first.severity == "high"
        # State-level errors still present, just no longer leading.
        components = [e.component for e in result.validation.errors[1:]]
        assert "source" in components
        assert "pipeline" in components

    def test_data_error_mirrors_leading_validation_message(self) -> None:
        """data.error and validation.errors[0].message must match.

        The two channels exist for backward compatibility with
        consumers that read either field. They must stay in sync so a
        consumer reading one cannot disagree with a consumer reading
        the other.
        """
        state = _empty_state()
        result = _failure_result(state, "rejection text")
        assert result.data["error"] == result.validation.errors[0].message

    def test_preserves_warnings_and_semantic_contracts(self) -> None:
        """Non-error fields on the input ValidationSummary survive prepending."""
        # state.validate() on an empty state yields no warnings/suggestions,
        # so this test asserts the ValidationSummary fields are reachable
        # and unchanged in shape after _failure_result wraps them.
        state = _empty_state()
        result = _failure_result(state, "x")
        assert result.validation.warnings == ()
        assert result.validation.suggestions == ()
        assert result.validation.semantic_contracts == ()
        assert result.validation.edge_contracts == ()


class TestToolResultSemanticContracts:
    """ToolResult.to_dict() must surface semantic_contracts.

    Every mutation tool (upsert_node, set_source, patch_*) returns a
    ToolResult; validation produced by state.validate() now carries
    semantic_contracts. Without exposing them in to_dict(), MCP clients
    only see the legacy errors/warnings fields and miss the structured
    plugin-declared contract records.
    """

    def test_tool_result_to_dict_includes_semantic_contracts(self) -> None:
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ")
        validation = state.validate()
        tr = ToolResult(
            success=True,
            updated_state=state,
            validation=validation,
            affected_nodes=(),
        )
        payload = tr.to_dict()
        assert "semantic_contracts" in payload["validation"]
        assert len(payload["validation"]["semantic_contracts"]) == 1
        contract = payload["validation"]["semantic_contracts"][0]
        assert contract["outcome"] == "conflict"
        assert contract["consumer_plugin"] == "line_explode"
        assert contract["producer_plugin"] == "web_scrape"
        assert contract["from_id"] == "scrape"
        assert contract["to_id"] == "explode"
        assert contract["requirement_code"] == "line_explode.source_field.line_framed_text"

    def test_tool_result_to_dict_emits_empty_list_for_no_contracts(self) -> None:
        """Surface parity: empty list when no contracts, not omitted."""
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=(),
        )
        d = result.to_dict()
        assert "semantic_contracts" in d["validation"]
        assert d["validation"]["semantic_contracts"] == []


class TestPreviewPipelineSemanticContracts:
    """_execute_preview_pipeline summary must include semantic_contracts."""

    def test_summary_includes_semantic_contracts(self) -> None:
        from elspeth.web.composer.tools import _execute_preview_pipeline
        from elspeth.web.composer.tools._common import ToolContext
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ")
        result = _execute_preview_pipeline({}, state, ToolContext(catalog=_mock_catalog()))
        assert "semantic_contracts" in result.data
        assert len(result.data["semantic_contracts"]) == 1
        assert result.data["semantic_contracts"][0]["outcome"] == "conflict"
        assert result.data["semantic_contracts"][0]["consumer_plugin"] == "line_explode"


class TestSetSource:
    def test_sets_source(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert _default_source(result.updated_state) is not None
        assert _default_source(result.updated_state).plugin == "csv"
        assert result.updated_state.version == 2
        assert "source" in result.affected_nodes

    def test_set_source_with_source_name_adds_named_source(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        first = execute_tool(
            "set_source",
            {
                "source_name": "customers",
                "plugin": "csv",
                "on_success": "customer_rows",
                "options": {"path": "/data/customers.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            state,
            catalog,
        )
        second = execute_tool(
            "set_source",
            {
                "source_name": "orders",
                "plugin": "json",
                "on_success": "order_rows",
                "options": {"path": "/data/orders.json", "schema": {"mode": "observed"}},
                "on_validation_failure": "bad_orders",
            },
            first.updated_state,
            catalog,
        )

        assert second.success is True
        assert tuple(second.updated_state.sources) == ("customers", "orders")
        assert second.updated_state.sources["orders"].plugin == "json"
        assert "source:orders" in second.affected_nodes

    @pytest.mark.parametrize("source_name", ("Orders", "on_success", " "))
    def test_set_source_invalid_source_name_raises_arg_error(self, source_name: str) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()

        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "set_source",
                {
                    "source_name": source_name,
                    "plugin": "csv",
                    "on_success": "rows",
                    "options": {"path": "/data/orders.csv", "schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                },
                state,
                catalog,
            )

        assert exc_info.value.argument == "source_name"
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_named_source_patch_and_clear_target_only_selected_source(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        customers = execute_tool(
            "set_source",
            {
                "source_name": "customers",
                "plugin": "csv",
                "on_success": "customer_rows",
                "options": {"path": "/data/customers.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            state,
            catalog,
        ).updated_state
        both = execute_tool(
            "set_source",
            {
                "source_name": "orders",
                "plugin": "json",
                "on_success": "order_rows",
                "options": {"path": "/data/orders.json", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            customers,
            catalog,
        ).updated_state

        patched = execute_tool(
            "patch_source_options",
            {"source_name": "orders", "patch": {"path": "/data/orders-patched.json"}},
            both,
            catalog,
        )
        cleared = execute_tool("clear_source", {"source_name": "customers"}, patched.updated_state, catalog)

        assert patched.success is True
        assert patched.updated_state.sources["orders"].options["path"] == "/data/orders-patched.json"
        assert patched.updated_state.sources["customers"].options["path"] == "/data/customers.csv"
        assert cleared.success is True
        assert tuple(cleared.updated_state.sources) == ("orders",)

    def test_on_validation_failure_accepts_sink_name(self) -> None:
        """on_validation_failure can be a sink name — not just 'discard'/'quarantine'.

        Regression guard: the tool schema must not constrain on_validation_failure
        to an enum. The runtime accepts any valid sink name for routing validation
        failures (e.g. 'bad_rows_sink'). If an enum constraint is re-added, the
        LLM cannot build source-level failsink routes.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "bad_rows_sink",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert _default_source(result.updated_state) is not None
        assert _default_source(result.updated_state).on_validation_failure == "bad_rows_sink"

    def test_unknown_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: foobar")
        result = execute_tool(
            "set_source",
            {
                "plugin": "foobar",
                "on_success": "t1",
                "options": {},
                "on_validation_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success is False
        assert _default_source(result.updated_state) is None  # unchanged
        assert result.updated_state.version == 1
        assert result.data is not None
        assert "foobar" in result.data["error"].lower()

    def test_set_source_rejects_manual_blob_ref_in_options(self) -> None:
        """Closes elspeth-07089fbaa3 (write defense, branch a).

        set_source must not accept ``blob_ref`` in options because it
        cannot enforce that the supplied ``path`` equals the bound blob's
        canonical ``storage_path``.  The canonical write path for blob-
        backed sources is ``set_source_from_blob``, which forces the path
        to ``BlobRecord.storage_path``.  Without this rejection, a caller
        (the LLM) could persist a wrong-shape path alongside a real
        blob_ref, breaking runtime path resolution.

        Bug-verification protocol (cf.
        ``tests/integration/pipeline/test_composer_runtime_agreement.py``
        module docstring lines 76-88): manually revert the
        ``"blob_ref" in options`` check in ``_execute_set_source``
        (src/elspeth/web/composer/tools.py) and confirm this test fails
        with ``result.success is True``.  Then restore.  This guards
        against the test passing both pre-fix and post-fix — a class of
        test theatre otherwise undetectable until a future regression
        slips through.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {
                    "blob_ref": "abc123",
                    "path": "data/blobs/abc123/tickets.csv",
                    "schema": {"mode": "observed"},
                },
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert result.success is False
        assert _default_source(result.updated_state) is None
        assert "set_source_from_blob" in result.data["error"]

    def test_set_source_rejects_manual_source_authoring_in_options(self) -> None:
        """Caller-supplied source_authoring must not bypass blob provenance stamping."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
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
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )

        assert result.success is False
        assert "source" not in result.updated_state.sources
        assert SOURCE_AUTHORING_KEY in result.data["error"]


class TestVfDestinationAdvisory:
    """Advisory note when on_validation_failure references an unknown output.

    The set_source tool schema accepts any string for on_validation_failure
    (not just 'discard'/'quarantine'). When the value doesn't match a
    configured output, ToolResult.data includes a note so the LLM can
    self-correct before pipeline validation fails at engine startup.
    """

    def test_set_source_unknown_vf_sink_includes_note(self) -> None:
        """Unknown on_validation_failure destination produces advisory note."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "nonexistent",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert result.data is not None
        assert "nonexistent" in result.data["note"]
        assert "output" in result.data["note"].lower()
        assert "discard" in result.data["note"]

    def test_set_source_discard_vf_no_note(self) -> None:
        """'discard' is a built-in value — no advisory needed."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert result.data is None

    def test_set_source_known_vf_sink_no_note(self) -> None:
        """When the named output exists, no advisory is needed."""
        state = _empty_state()
        catalog = _mock_catalog()
        # First create the output that on_validation_failure will reference.
        r1 = execute_tool(
            "set_output",
            {
                "sink_name": "quarantine",
                "plugin": "csv",
                "options": {"path": "/data/quarantine.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert r1.success is True
        # Now set source referencing the existing output.
        r2 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            r1.updated_state,
            catalog,
        )
        assert r2.success is True
        assert r2.data is None

    def test_set_pipeline_unknown_vf_sink_includes_note(self) -> None:
        """set_pipeline with unknown on_validation_failure produces advisory."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        # Change on_validation_failure to a name that doesn't match any output.
        args["source"]["on_validation_failure"] = "typo_sink"
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is True
        assert result.data is not None
        assert "typo_sink" in result.data["note"]

    def test_set_pipeline_vf_matches_output_no_note(self) -> None:
        """set_pipeline with on_validation_failure matching an output — no note."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        # Add a "quarantine" output and reference it from on_validation_failure.
        args["outputs"].append(
            {
                "sink_name": "quarantine",
                "plugin": "csv",
                "options": {"path": "/data/quarantine.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
        )
        args["source"]["on_validation_failure"] = "quarantine"
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is True
        assert result.data is None


class TestUpsertNode:
    def test_adds_new_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "source_out",
                "on_success": "main",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        assert result.success is True
        assert len(result.updated_state.nodes) == 1
        assert "t1" in result.affected_nodes

    def test_allows_llm_consuming_web_scrape_without_prompt_shield_as_advisory(self) -> None:
        state = CompositionState(
            source=None,
            nodes=(
                NodeSpec(
                    id="fetch_pages",
                    node_type="transform",
                    plugin="web_scrape",
                    input="rows",
                    on_success="scraped_rows",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "url_field": "url",
                        "content_field": "content",
                        "http": {
                            "abuse_contact": "ops@example.com",
                            "scraping_reason": "technical evaluation",
                            "allowed_hosts": ["example.com"],
                        },
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
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        catalog = _mock_catalog()

        result = execute_tool(
            "upsert_node",
            {
                "id": "summarise_pages",
                "node_type": "transform",
                "plugin": "llm",
                "input": "scraped_rows",
                "on_success": "summaries",
                "on_error": "discard",
                "options": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    "prompt_template": "Summarise {{ row.content }}.",
                    "schema": {"mode": "observed"},
                },
            },
            state,
            catalog,
        )

        # Advisory, not blocking: the missing prompt-shield surfaces as a
        # non-blocking validation warning rather than rejecting the upsert.
        assert result.success is True
        warning_text = " ".join(w.message for w in result.updated_state.validate().warnings)
        assert PROMPT_SHIELD_USER_TERM in warning_text
        assert "continuing without it is allowed" in warning_text

    def test_replaces_existing_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result1 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        result2 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "new_in",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
            },
            result1.updated_state,
            catalog,
        )
        assert result2.success is True
        assert len(result2.updated_state.nodes) == 1
        assert result2.updated_state.nodes[0].input == "new_in"

    def test_gate_node_no_plugin_validation(self) -> None:
        """Gates don't have plugins — should not validate against catalog."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": "row['x'] > 0",
                "routes": {"true": "s1", "false": "s2"},
            },
            state,
            catalog,
        )
        assert result.success is True
        catalog.get_schema.assert_not_called()

    def test_upsert_node_unknown_transform_plugin_fails(self) -> None:
        """W-4B-1: LLM hallucinates a transform plugin name."""
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: nonexistent_xyz")
        result = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "nonexistent_xyz",
                "input": "in",
                "on_success": "out",
                "options": {},
            },
            state,
            catalog,
        )
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged

    def test_gate_injection_condition_rejected(self) -> None:
        """upsert_node rejects gate with injection attempt in condition."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": "__import__('os').system('rm -rf /')",
                "routes": {"true": "s1", "false": "s2"},
            },
            state,
            catalog,
        )
        assert result.success is False
        assert "Forbidden construct" in result.data["error"]
        assert result.updated_state.version == 1  # unchanged

    def test_gate_malformed_condition_rejected(self) -> None:
        """upsert_node rejects gate with syntactically invalid condition."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": "row['x'] >== 5",
                "routes": {"true": "s1", "false": "s2"},
            },
            state,
            catalog,
        )
        assert result.success is False
        assert "Invalid gate condition syntax" in result.data["error"]
        assert result.updated_state.version == 1

    def test_gate_boolean_condition_custom_labels_rejected(self) -> None:
        """upsert_node rejects a boolean gate condition with non-true/false labels.

        Mirrors GateSettings.validate_boolean_routes at the tool boundary so the
        composer does not green-light a pipeline runtime config later rejects.
        Regression for elspeth-08e17b9253.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": "row['x'] > 0",
                "routes": {"high": "s1", "low": "s2"},
            },
            state,
            catalog,
        )
        assert result.success is False
        assert "boolean condition" in result.data["error"]
        assert result.updated_state.version == 1

    def test_gate_numeric_condition_rejected(self) -> None:
        """upsert_node rejects a provably-numeric gate condition (never a route label).

        Regression for elspeth-08e17b9253.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": "row['x'] + 1",
                "routes": {"a": "s1"},
            },
            state,
            catalog,
        )
        assert result.success is False
        assert "numeric value" in result.data["error"]
        assert result.updated_state.version == 1

    def test_gate_eval_call_rejected(self) -> None:
        """upsert_node rejects eval() in condition (forbidden function call)."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": "eval('row[\"x\"]')",
                "routes": {"true": "s1", "false": "s2"},
            },
            state,
            catalog,
        )
        assert result.success is False
        assert "Forbidden construct" in result.data["error"]

    def test_gate_valid_condition_accepted(self) -> None:
        """upsert_node accepts gate with well-formed condition."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": "row['score'] >= 0.85 and row.get('status') is not None",
                "routes": {"true": "s1", "false": "s2"},
            },
            state,
            catalog,
        )
        assert result.success is True
        assert len(result.updated_state.nodes) == 1

    def test_aggregation_end_of_source_condition_rejected(self) -> None:
        """upsert_node rejects end_of_source in the aggregation condition slot."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "agg1",
                "node_type": "aggregation",
                "plugin": "batch_stats",
                "input": "source_out",
                "on_success": "main",
                "options": {"schema": {"mode": "observed"}, "value_field": "amount"},
                "trigger": {"condition": "end_of_source"},
            },
            state,
            catalog,
        )

        assert result.success is False
        assert "end_of_source" in result.data["error"]
        assert result.updated_state.version == 1

    def test_gate_none_condition_not_validated(self) -> None:
        """upsert_node with condition=None skips expression validation.

        Presence validation is the job of CompositionState.validate(), not
        the upsert_node handler. This test ensures we don't crash on None.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "in",
                "on_success": None,
                "options": {},
                "condition": None,
                "routes": {"true": "s1", "false": "s2"},
            },
            state,
            catalog,
        )
        # Succeeds at tool level; validate() will flag missing condition
        assert result.success is True

    def test_transform_with_condition_skips_expression_validation(self) -> None:
        """Non-gate nodes with a condition field don't trigger expression validation.

        Only gates have expressions; a transform with a stray condition field
        is a structural error caught by validate(), not an expression error.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
                "condition": "this is not python!!!",
            },
            state,
            catalog,
        )
        # The invalid syntax is irrelevant for a transform node — not validated
        assert result.success is True


class TestUpsertEdge:
    def _catalog_with_llm_and_json(self) -> MagicMock:
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="llm",
                description="LLM transform",
                plugin_type="transform",
                config_fields=[],
            ),
        ]
        catalog.list_sinks.return_value = [
            *catalog.list_sinks.return_value,
            PluginSummary(
                name="json",
                description="JSON file sink",
                plugin_type="sink",
                config_fields=[],
            ),
        ]
        return catalog

    def test_adds_edge(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_edge",
            {
                "id": "e1",
                "from_node": "source",
                "to_node": "t1",
                "edge_type": "on_success",
                "label": None,
            },
            state,
            catalog,
        )
        assert result.success is True
        assert len(result.updated_state.edges) == 1
        assert "source" in result.affected_nodes
        assert "t1" in result.affected_nodes

    def test_edge_to_output_syncs_node_on_success(self) -> None:
        """Edge from node to output updates node's on_success to output name."""
        state = _empty_state()
        catalog = _mock_catalog()
        # Add a node with on_success pointing elsewhere
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "old_stream",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        # Add an output
        r2 = execute_tool(
            "set_output",
            {
                "sink_name": "csv_out",
                "plugin": "csv",
                "options": {"path": "/data/outputs/output.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r1.updated_state,
            catalog,
        )
        # Add edge from node to output with on_success type
        r3 = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "t1", "to_node": "csv_out", "edge_type": "on_success"},
            r2.updated_state,
            catalog,
        )
        assert r3.success is True
        node = next(n for n in r3.updated_state.nodes if n.id == "t1")
        assert node.on_success == "csv_out"

    def test_edge_to_output_syncs_node_on_error(self) -> None:
        """Edge from node to output with on_error updates node's on_error."""
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "set_output",
            {
                "sink_name": "err_out",
                "plugin": "csv",
                "options": {"path": "/data/outputs/output.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r1.updated_state,
            catalog,
        )
        r3 = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "t1", "to_node": "err_out", "edge_type": "on_error"},
            r2.updated_state,
            catalog,
        )
        assert r3.success is True
        node = next(n for n in r3.updated_state.nodes if n.id == "t1")
        assert node.on_error == "err_out"

    def test_upsert_edge_adds_llm_failure_sink_on_error_without_rebuilding_pipeline(self) -> None:
        """An existing LLM node can be routed to a failure sink via upsert_edge."""
        state = _empty_state()
        catalog = self._catalog_with_llm_and_json()
        with_node = execute_tool(
            "upsert_node",
            {
                "id": "judge_layers",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "judged_layers",
                "on_error": "discard",
                "options": _llm_options_with_api_key({"secret_ref": "OPENROUTER_API_KEY"}),
            },
            state,
            catalog,
        )
        assert with_node.success is True
        with_failure_sink = execute_tool(
            "set_output",
            {
                "sink_name": "llm_failures",
                "plugin": "json",
                "options": {
                    "path": "/data/outputs/magic_comp_rules_layers_failures.json",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            with_node.updated_state,
            catalog,
            data_dir="/data",
        )
        assert with_failure_sink.success is True

        result = execute_tool(
            "upsert_edge",
            {
                "id": "e_judge_layers_error",
                "from_node": "judge_layers",
                "to_node": "llm_failures",
                "edge_type": "on_error",
                "label": "LLM failures",
            },
            with_failure_sink.updated_state,
            catalog,
        )

        assert result.success is True
        assert result.updated_state.nodes[0].on_error == "llm_failures"
        assert result.updated_state.nodes[0].options == with_failure_sink.updated_state.nodes[0].options
        assert result.updated_state.edges[-1].edge_type == "on_error"

    @pytest.mark.parametrize(
        ("edge_type", "expected_routes"),
        [
            ("route_true", {"true": "main", "false": "old_false"}),
            ("route_false", {"true": "old_true", "false": "main"}),
        ],
    )
    def test_gate_route_edge_to_output_syncs_gate_routes(
        self,
        edge_type: str,
        expected_routes: dict[str, str],
    ) -> None:
        """Gate route edges to sinks must update the gate's runtime routes."""
        state = _empty_state().with_node(
            NodeSpec(
                id="g1",
                node_type="gate",
                plugin=None,
                input="in",
                on_success=None,
                on_error=None,
                options={},
                condition="row['flag']",
                routes={"true": "old_true", "false": "old_false"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        catalog = _mock_catalog()
        with_output = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )

        result = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "g1", "to_node": "main", "edge_type": edge_type},
            with_output.updated_state,
            catalog,
        )

        assert result.success is True
        gate = next(n for n in result.updated_state.nodes if n.id == "g1")
        assert dict(gate.routes or {}) == expected_routes

    def test_gate_fork_edge_to_output_syncs_gate_fork_to(self) -> None:
        """Fork edges to sinks must update the gate's runtime fork destinations."""
        state = _empty_state().with_node(
            NodeSpec(
                id="g1",
                node_type="gate",
                plugin=None,
                input="in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("existing",),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        catalog = _mock_catalog()
        with_output = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )

        result = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "g1", "to_node": "main", "edge_type": "fork"},
            with_output.updated_state,
            catalog,
        )

        assert result.success is True
        gate = next(n for n in result.updated_state.nodes if n.id == "g1")
        assert gate.fork_to == ("existing", "main")

    def test_route_edge_from_transform_to_output_is_rejected(self) -> None:
        """Only gates can use route_true/route_false/fork sink edges."""
        state = _empty_state()
        catalog = _mock_catalog()
        with_node = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        with_output = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            with_node.updated_state,
            catalog,
        )

        result = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "t1", "to_node": "main", "edge_type": "route_true"},
            with_output.updated_state,
            catalog,
        )

        assert result.success is False
        assert "gate" in result.data["error"].lower()

    def test_edge_to_output_syncs_source_on_success(self) -> None:
        """Edge from source to output updates source's on_success.

        set_source is promoted to a Pydantic argument model with
        extra='forbid'; all four required fields must be supplied
        (Task 4).
        """
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "old_stream",
                "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/outputs/output.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r1.updated_state,
            catalog,
        )
        r3 = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "source", "to_node": "main", "edge_type": "on_success"},
            r2.updated_state,
            catalog,
        )
        assert r3.success is True
        assert _default_source(r3.updated_state) is not None
        assert _default_source(r3.updated_state).on_success == "main"

    def test_edge_to_node_does_not_sync(self) -> None:
        """Edge from node to another node does NOT change connection fields."""
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "stream_a",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "upsert_node",
            {
                "id": "t2",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "stream_a",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
            },
            r1.updated_state,
            catalog,
        )
        r3 = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "t1", "to_node": "t2", "edge_type": "on_success"},
            r2.updated_state,
            catalog,
        )
        assert r3.success is True
        node = next(n for n in r3.updated_state.nodes if n.id == "t1")
        assert node.on_success == "stream_a"  # unchanged

    def test_edge_to_output_already_matching_is_noop(self) -> None:
        """Edge to output where on_success already matches does not double-bump version."""
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "csv_out",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "set_output",
            {
                "sink_name": "csv_out",
                "plugin": "csv",
                "options": {"path": "/data/outputs/output.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r1.updated_state,
            catalog,
        )
        v_before = r2.updated_state.version
        r3 = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "t1", "to_node": "csv_out", "edge_type": "on_success"},
            r2.updated_state,
            catalog,
        )
        assert r3.success is True
        node = next(n for n in r3.updated_state.nodes if n.id == "t1")
        assert node.on_success == "csv_out"
        # with_edge bumps version once; with_node should NOT be called
        assert r3.updated_state.version == v_before + 1


class TestRemoveNode:
    def test_removes_node_and_edges(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # Add a node and an edge to it
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "in",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "upsert_edge",
            {
                "id": "e1",
                "from_node": "source",
                "to_node": "t1",
                "edge_type": "on_success",
                "label": None,
            },
            r1.updated_state,
            catalog,
        )
        # Remove the node — edge should also be removed
        r3 = execute_tool("remove_node", {"id": "t1"}, r2.updated_state, catalog)
        assert r3.success is True
        assert len(r3.updated_state.nodes) == 0
        assert len(r3.updated_state.edges) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_node", {"id": "nope"}, state, catalog)
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged


class TestRemoveEdge:
    def test_removes_edge(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "upsert_edge",
            {
                "id": "e1",
                "from_node": "source",
                "to_node": "t1",
                "edge_type": "on_success",
                "label": None,
            },
            state,
            catalog,
        )
        r2 = execute_tool("remove_edge", {"id": "e1"}, r1.updated_state, catalog)
        assert r2.success is True
        assert len(r2.updated_state.edges) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_edge", {"id": "nope"}, state, catalog)
        assert result.success is False


class TestSetMetadata:
    def test_partial_update(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_metadata",
            {"patch": {"name": "My Pipeline"}},
            state,
            catalog,
        )
        assert result.success is True
        assert result.updated_state.metadata.name == "My Pipeline"
        assert result.updated_state.metadata.description == ""  # preserved
        assert result.affected_nodes == ()  # metadata doesn't affect nodes

    def test_missing_patch_key_raises(self) -> None:
        """LLM omits required 'patch' key — route as Tier-3 argument error."""
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError):
            execute_tool("set_metadata", {"name": "Oops"}, state, catalog)


class TestLegacyMutationArgumentGuards:
    @pytest.mark.parametrize(
        ("tool_name", "arguments"),
        [
            ("upsert_node", {}),
            ("upsert_edge", {}),
            ("remove_node", {}),
            ("remove_edge", {}),
            ("set_output", {}),
            ("remove_output", {}),
        ],
    )
    def test_missing_required_fields_raise_tool_argument_error(self, tool_name: str, arguments: dict[str, Any]) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(tool_name, arguments, _empty_state(), _mock_catalog())

        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_upsert_node_rejects_unknown_node_type_before_state_mutation(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "upsert_node",
                {"id": "x", "node_type": "bogus", "input": "in"},
                _empty_state(),
                _mock_catalog(),
            )

        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_upsert_gate_rejects_non_string_condition_as_argument_error(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "upsert_node",
                {"id": "gate1", "node_type": "gate", "input": "rows", "condition": 42},
                _empty_state(),
                _mock_catalog(),
            )

        assert isinstance(exc_info.value.__cause__, PydanticValidationError)


class TestSetOutput:
    def test_adds_output(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert len(result.updated_state.outputs) == 1
        assert result.updated_state.outputs[0].name == "main"
        assert result.updated_state.outputs[0].plugin == "csv"
        assert result.updated_state.version == 2
        assert "main" in result.affected_nodes

    def test_data_dir_file_sink_requires_collision_policy(self) -> None:
        """Runnable web-composer file sinks must make output collision behavior explicit."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
            data_dir="/data",
        )

        assert result.success is False
        assert "collision_policy" in result.data["error"]

    def test_data_dir_file_sink_accepts_explicit_collision_policy(self) -> None:
        """The composer accepts file sinks once the LLM chooses the collision policy."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {
                    "path": "/data/outputs/out.csv",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            state,
            catalog,
            data_dir="/data",
        )

        assert result.success is True
        assert result.updated_state.outputs[0].options["collision_policy"] == "auto_increment"

    def test_replaces_existing_output(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/new.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "quarantine",
            },
            r1.updated_state,
            catalog,
        )
        assert r2.success is True
        assert len(r2.updated_state.outputs) == 1
        assert r2.updated_state.outputs[0].on_write_failure == "quarantine"

    def test_on_write_failure_accepts_sink_name_for_failsink_routing(self) -> None:
        """on_write_failure can be a sink name — not just 'discard'/'quarantine'.

        Regression guard: the tool schema must not constrain on_write_failure to
        an enum. The skill document instructs LLMs to set it to a sink name (e.g.
        'results_failures') to wire automatic failsink pipelines. If an enum
        constraint is re-added, the LLM cannot build failsink routes.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "results_failures",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert result.updated_state.outputs[0].on_write_failure == "results_failures"

    def test_unknown_sink_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: foobar")
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "foobar",
                "options": {},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged

    def test_set_output_rejects_secret_ref_in_non_credential_field(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {
                    "path": {"secret_ref": "ANY_SECRET"},
                    "schema": {"mode": "observed"},
                },
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )

        assert result.success is False
        assert result.updated_state is state
        assert "csv" in result.data["error"]
        assert "path" in result.data["error"]
        assert "ANY_SECRET" in result.data["error"]
        assert "only credential-bearing fields" in result.data["error"]


class TestRemoveOutput:
    def test_removes_output(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/outputs/output.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        r2 = execute_tool("remove_output", {"sink_name": "main"}, r1.updated_state, catalog)
        assert r2.success is True
        assert len(r2.updated_state.outputs) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_output", {"sink_name": "nope"}, state, catalog)
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged


class TestSetSourcePathSecurity:
    """S2: Source path allowlist — paths must be under {data_dir}/blobs/."""

    def test_path_under_blobs_succeeds(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is True

    def test_path_outside_blobs_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/etc/passwd"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False
        assert "path" in result.data["error"].lower()

    def test_traversal_attack_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/blobs/../../etc/passwd"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_file_key_also_validated(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"file": "/tmp/evil.csv"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_file_key_traversal_via_blobs_prefix_fails(self) -> None:
        """W-4B-2: file key traversal starting from blobs prefix."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"file": "/data/blobs/../../etc/passwd"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_no_path_key_skips_s2_but_fails_prevalidation(self) -> None:
        """Source options without path/file keys are not subject to S2 path security.

        S2 only checks path/file keys for directory traversal — absent keys are not
        rejected. However, pre-validation (Pydantic) correctly rejects the call because
        csv source requires 'path'. The failure comes from pre-validation, not S2.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        # Pydantic catches the missing required 'path' field
        assert result.success is False
        # Error is from pre-validation (path required), not S2 (traversal / allowed dir)
        assert "path" in result.data["error"]
        assert "traversal" not in result.data["error"].lower()
        assert "allowed" not in result.data["error"].lower()

    def test_relative_path_resolves_against_data_dir(self) -> None:
        """blobs/input.csv should resolve under {data_dir}/blobs/."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "blobs/input.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is True

    def test_relative_traversal_still_blocked(self) -> None:
        """../etc/passwd relative to data_dir must still be blocked."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "../etc/passwd"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False


class TestDiscoveryTools:
    def test_list_sources_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_sources", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_sources.assert_called_once()

    def test_list_transforms_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_transforms", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_transforms.assert_called_once()

    def test_list_sinks_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_sinks", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_sinks.assert_called_once()

    def test_get_plugin_schema_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "csv"},
            _empty_state(),
            catalog,
        )
        assert result.success is True
        catalog.get_schema.assert_called_once_with("source", "csv")

    def test_list_transforms_hides_secret_required_plugin_without_declared_candidate(self) -> None:
        from elspeth.contracts.secrets import SecretInventoryItem

        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="azure_prompt_shield",
                description="Prompt injection shield",
                plugin_type="transform",
                config_fields=[
                    ConfigFieldSummary(name="api_key", type="string", required=True),
                    ConfigFieldSummary(name="endpoint", type="string", required=True),
                ],
                secret_requirements=(PluginSecretRequirement(field="api_key", candidates=("AZURE_CONTENT_SAFETY_KEY",)),),
            ),
        ]
        secret_service = MagicMock()
        secret_service.list_refs.return_value = [
            SecretInventoryItem(name="OPENROUTER_API_KEY", scope="user", available=True),
        ]
        secret_service.has_ref.side_effect = lambda _user_id, name: name == "OPENROUTER_API_KEY"

        result = execute_tool(
            "list_transforms",
            {},
            _empty_state(),
            catalog,
            secret_service=secret_service,
            user_id="test-user",
        )

        assert result.success is True
        names = {item.name for item in result.data}
        assert "passthrough" in names
        assert "azure_prompt_shield" not in names

    def test_list_transforms_shows_secret_required_plugin_when_declared_candidate_available(self) -> None:
        from elspeth.contracts.secrets import SecretInventoryItem

        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="azure_prompt_shield",
                description="Prompt injection shield",
                plugin_type="transform",
                config_fields=[
                    ConfigFieldSummary(name="api_key", type="string", required=True),
                    ConfigFieldSummary(name="endpoint", type="string", required=True),
                ],
                secret_requirements=(PluginSecretRequirement(field="api_key", candidates=("AZURE_CONTENT_SAFETY_KEY",)),),
            ),
        ]
        secret_service = MagicMock()
        secret_service.list_refs.return_value = [
            SecretInventoryItem(name="AZURE_CONTENT_SAFETY_KEY", scope="server", available=True),
        ]
        secret_service.has_ref.side_effect = lambda _user_id, name: name == "AZURE_CONTENT_SAFETY_KEY"

        result = execute_tool(
            "list_transforms",
            {},
            _empty_state(),
            catalog,
            secret_service=secret_service,
            user_id="test-user",
        )

        assert result.success is True
        names = {item.name for item in result.data}
        assert "azure_prompt_shield" in names

    def test_get_plugin_schema_rejects_secret_required_plugin_without_declared_candidate(self) -> None:
        catalog = _mock_catalog()
        catalog.get_schema.return_value = PluginSchemaInfo(
            name="azure_prompt_shield",
            plugin_type="transform",
            description="Prompt injection shield",
            json_schema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string"},
                    "endpoint": {"type": "string"},
                },
                "required": ["api_key", "endpoint"],
            },
            knob_schema={"fields": []},
            secret_requirements=(PluginSecretRequirement(field="api_key", candidates=("AZURE_CONTENT_SAFETY_KEY",)),),
        )
        secret_service = MagicMock()
        secret_service.has_ref.return_value = False

        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "transform", "name": "azure_prompt_shield"},
            _empty_state(),
            catalog,
            secret_service=secret_service,
            user_id="test-user",
        )

        assert result.success is False
        assert "azure_prompt_shield" in result.data["error"]
        assert "AZURE_CONTENT_SAFETY_KEY" in result.data["error"]

    def test_get_plugin_assistance_rejects_secret_required_plugin_without_declared_candidate(self) -> None:
        catalog = _mock_catalog()
        catalog.get_schema.return_value = PluginSchemaInfo(
            name="azure_prompt_shield",
            plugin_type="transform",
            description="Prompt injection shield",
            json_schema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string"},
                    "endpoint": {"type": "string"},
                },
                "required": ["api_key", "endpoint"],
            },
            knob_schema={"fields": []},
            secret_requirements=(PluginSecretRequirement(field="api_key", candidates=("AZURE_CONTENT_SAFETY_KEY",)),),
        )
        secret_service = MagicMock()
        secret_service.has_ref.return_value = False

        result = execute_tool(
            "get_plugin_assistance",
            {"plugin_type": "transform", "plugin_name": "azure_prompt_shield"},
            _empty_state(),
            catalog,
            secret_service=secret_service,
            user_id="test-user",
        )

        assert result.success is False
        assert "azure_prompt_shield" in result.data["error"]
        assert "AZURE_CONTENT_SAFETY_KEY" in result.data["error"]

    def test_get_expression_grammar_is_static(self) -> None:
        grammar = get_expression_grammar()
        assert "row" in grammar
        assert isinstance(grammar, str)


class TestToolDefinitions:
    def test_all_have_json_schema(self) -> None:
        for defn in get_tool_definitions():
            assert "name" in defn
            assert "description" in defn
            assert "parameters" in defn

    def test_array_schemas_declare_items(self) -> None:
        """Provider contract: every JSON-Schema array must declare items.

        OpenAI rejects tool schemas with bare ``{"type": "array"}`` at
        request validation time. This walks the full composer tool surface so
        one malformed nested schema cannot make every new session fail before
        the model sees the prompt.
        """
        for defn in get_tool_definitions():
            self._assert_arrays_have_items(defn["parameters"], defn["name"], ("parameters",))

    def test_on_validation_failure_has_no_enum_constraint(self) -> None:
        """Regression: on_validation_failure must accept any sink name, not just enum values.

        The runtime accepts 'discard' or any valid sink name for source
        validation failure routing.  A hard-coded enum blocks LLMs from
        building source-level failsink pipelines.
        """
        for defn in get_tool_definitions():
            self._assert_no_enum_on_validation_failure(defn.get("parameters", {}), defn["name"])

    def test_on_validation_failure_descriptions_match_runtime_contract(self) -> None:
        """Tool docs must not advertise a non-existent built-in quarantine sink."""
        descriptions: list[tuple[str, str]] = []
        for defn in get_tool_definitions():
            self._collect_on_validation_failure_descriptions(defn.get("parameters", {}), defn["name"], descriptions)

        assert descriptions, "expected at least one on_validation_failure schema description"
        for tool_name, description in descriptions:
            lowered = description.lower()
            assert "built-in quarantine" not in lowered, tool_name
            assert "discard" in description, tool_name
            assert "Any other value, including 'quarantine', must match a configured output/sink name." in description, tool_name

    @pytest.mark.parametrize(
        "tool_name",
        ["list_blobs", "list_composer_blobs", "get_blob_metadata", "inspect_source"],
    )
    def test_blob_discovery_schemas_forbid_extra_arguments(self, tool_name: str) -> None:
        """Blob discovery tools persist arguments; their LLM-facing schemas must reject extra keys."""
        definition = next(defn for defn in get_tool_definitions() if defn["name"] == tool_name)

        assert definition["parameters"]["additionalProperties"] is False

    def test_upsert_node_trigger_schema_documents_end_of_source_only_shape(self) -> None:
        """Aggregation trigger schema must expose the end-of-source-only shape."""
        upsert_node = next(defn for defn in get_tool_definitions() if defn["name"] == "upsert_node")
        trigger_schema = upsert_node["parameters"]["properties"]["trigger"]

        assert "null" in trigger_schema["type"]
        assert trigger_schema["additionalProperties"] is False
        assert set(trigger_schema["properties"]) == {"count", "timeout_seconds", "condition"}
        assert "end-of-source-only" in trigger_schema["description"]
        assert "do not use end_of_source" in trigger_schema["properties"]["condition"]["description"]

    def test_upsert_node_expected_output_count_warns_about_group_by(self) -> None:
        """Aggregation schema must not steer grouped rollups toward fixed cardinality."""
        upsert_node = next(defn for defn in get_tool_definitions() if defn["name"] == "upsert_node")
        expected_schema = upsert_node["parameters"]["properties"]["expected_output_count"]

        assert "omit when output count depends on group_by" in expected_schema["description"]

    def test_upsert_edge_schema_includes_llm_failure_sink_routing_example(self) -> None:
        """The tool surface must show how to patch LLM failure routing."""
        upsert_edge = next(defn for defn in get_tool_definitions() if defn["name"] == "upsert_edge")
        examples = upsert_edge["parameters"]["examples"]

        assert {
            "id": "e_judge_layers_error",
            "from_node": "judge_layers",
            "to_node": "llm_failures",
            "edge_type": "on_error",
            "label": "LLM failures",
        } in examples

    def test_get_pipeline_state_component_schema_documents_full_state_aliases(self) -> None:
        """Tool schema must expose full-state spellings instead of relying only on omission."""
        get_pipeline_state = next(defn for defn in get_tool_definitions() if defn["name"] == "get_pipeline_state")
        component_schema = get_pipeline_state["parameters"]["properties"]["component"]

        assert "Accepted full-state aliases" in component_schema["description"]
        assert "full" in component_schema["description"]
        assert "all" in component_schema["description"]
        assert "pipeline" in component_schema["description"]
        assert "empty string" in component_schema["description"]

    def _assert_no_enum_on_validation_failure(self, schema: object, tool_name: str) -> None:
        """Recursively walk a JSON schema and assert no on_validation_failure has enum."""
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key == "on_validation_failure" and isinstance(value, dict):
                    assert "enum" not in value, (
                        f"Tool {tool_name!r} constrains on_validation_failure to enum {value.get('enum')} — runtime accepts any sink name"
                    )
                elif isinstance(value, (dict, list)):
                    self._assert_no_enum_on_validation_failure(value, tool_name)
        elif isinstance(schema, list):
            for item in schema:
                self._assert_no_enum_on_validation_failure(item, tool_name)

    def _collect_on_validation_failure_descriptions(
        self,
        schema: object,
        tool_name: str,
        descriptions: list[tuple[str, str]],
    ) -> None:
        """Collect on_validation_failure descriptions from nested tool schemas."""
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key == "on_validation_failure" and isinstance(value, dict):
                    description = value.get("description")
                    assert isinstance(description, str), f"Tool {tool_name!r} on_validation_failure lacks description"
                    descriptions.append((tool_name, description))
                elif isinstance(value, (dict, list)):
                    self._collect_on_validation_failure_descriptions(value, tool_name, descriptions)
        elif isinstance(schema, list):
            for item in schema:
                self._collect_on_validation_failure_descriptions(item, tool_name, descriptions)

    def _assert_arrays_have_items(self, schema: object, tool_name: str, path: tuple[str, ...]) -> None:
        """Recursively walk a JSON schema and assert all arrays declare items."""
        if isinstance(schema, dict):
            schema_type = schema.get("type")
            has_array_type = schema_type == "array" or (isinstance(schema_type, list) and "array" in schema_type)
            assert not has_array_type or "items" in schema, f"Tool {tool_name!r} has array schema without items at {'.'.join(path)}"
            for key, value in schema.items():
                if isinstance(value, (dict, list)):
                    self._assert_arrays_have_items(value, tool_name, (*path, str(key)))
        elif isinstance(schema, list):
            for index, item in enumerate(schema):
                self._assert_arrays_have_items(item, tool_name, (*path, f"[{index}]"))


class TestToolResultValidation:
    def test_mutation_includes_validation(self) -> None:
        """Every mutation tool result includes validation summary."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert result.validation is not None
        # Source is set but no sinks — should have validation error
        assert not result.validation.is_valid
        assert any("No sinks" in e.message for e in result.validation.errors)


class TestComputeValidationDelta:
    """Tests for _compute_validation_delta identity semantics."""

    def test_same_message_different_component_not_collapsed(self) -> None:
        """Two entries with identical message but different component are distinct."""
        before = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("node:a", "Configuration incomplete.", "high"),),
        )
        after = ValidationSummary(
            is_valid=False,
            errors=(
                ValidationEntry("node:a", "Configuration incomplete.", "high"),
                ValidationEntry("node:b", "Configuration incomplete.", "high"),
            ),
        )
        delta = _compute_validation_delta(before, after)
        # node:b is genuinely new — must appear in new_errors
        assert len(delta["new_errors"]) == 1
        assert delta["new_errors"][0]["component"] == "node:b"
        assert delta["resolved_errors"] == []

    def test_same_component_same_message_not_duplicated(self) -> None:
        """Identical (component, message) across before/after is not new."""
        entry = ValidationEntry("source", "No source configured.", "high")
        before = ValidationSummary(is_valid=False, errors=(entry,))
        after = ValidationSummary(is_valid=False, errors=(entry,))
        delta = _compute_validation_delta(before, after)
        assert delta["new_errors"] == []
        assert delta["resolved_errors"] == []

    def test_resolved_entry_uses_component_identity(self) -> None:
        """An entry resolved for one component doesn't mask another."""
        before = ValidationSummary(
            is_valid=False,
            errors=(
                ValidationEntry("node:a", "Missing field.", "high"),
                ValidationEntry("node:b", "Missing field.", "high"),
            ),
        )
        after = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("node:b", "Missing field.", "high"),),
        )
        delta = _compute_validation_delta(before, after)
        assert len(delta["resolved_errors"]) == 1
        assert delta["resolved_errors"][0]["component"] == "node:a"
        assert delta["new_errors"] == []

    def test_warning_delta_uses_component_identity(self) -> None:
        """Warnings also use (component, message) identity."""
        before = ValidationSummary(
            is_valid=True,
            errors=(),
            warnings=(ValidationEntry("output:main", "No path configured.", "medium"),),
        )
        after = ValidationSummary(
            is_valid=True,
            errors=(),
            warnings=(
                ValidationEntry("output:main", "No path configured.", "medium"),
                ValidationEntry("output:backup", "No path configured.", "medium"),
            ),
        )
        delta = _compute_validation_delta(before, after)
        assert len(delta["new_warnings"]) == 1
        assert delta["new_warnings"][0]["component"] == "output:backup"

    def test_both_empty_yields_empty_delta(self) -> None:
        """Two empty validation states produce an all-empty delta."""
        before = ValidationSummary(is_valid=True, errors=(), warnings=())
        after = ValidationSummary(is_valid=True, errors=(), warnings=())
        delta = _compute_validation_delta(before, after)
        assert delta == {
            "new_errors": [],
            "resolved_errors": [],
            "new_warnings": [],
            "resolved_warnings": [],
        }

    def test_empty_before_makes_all_after_new(self) -> None:
        """When before is empty, every entry in after is new."""
        before = ValidationSummary(is_valid=True, errors=(), warnings=())
        after = ValidationSummary(
            is_valid=False,
            errors=(
                ValidationEntry("node:x", "Bad config.", "high"),
                ValidationEntry("source", "Missing field.", "high"),
            ),
            warnings=(ValidationEntry("output:main", "Slow path.", "medium"),),
        )
        delta = _compute_validation_delta(before, after)
        assert len(delta["new_errors"]) == 2
        assert len(delta["new_warnings"]) == 1
        assert delta["resolved_errors"] == []
        assert delta["resolved_warnings"] == []

    def test_empty_after_makes_all_before_resolved(self) -> None:
        """When after is empty, every entry in before is resolved."""
        before = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("node:a", "Missing plugin.", "high"),),
            warnings=(ValidationEntry("output:main", "No path.", "medium"),),
        )
        after = ValidationSummary(is_valid=True, errors=(), warnings=())
        delta = _compute_validation_delta(before, after)
        assert len(delta["resolved_errors"]) == 1
        assert delta["resolved_errors"][0]["component"] == "node:a"
        assert len(delta["resolved_warnings"]) == 1
        assert delta["resolved_warnings"][0]["component"] == "output:main"
        assert delta["new_errors"] == []
        assert delta["new_warnings"] == []

    def test_mixed_errors_and_warnings_independent(self) -> None:
        """Error and warning deltas are computed independently."""
        shared_entry = ValidationEntry("node:a", "Problem.", "high")
        before = ValidationSummary(
            is_valid=False,
            errors=(shared_entry,),
            warnings=(ValidationEntry("source", "Old warning.", "medium"),),
        )
        after = ValidationSummary(
            is_valid=False,
            errors=(shared_entry,),
            warnings=(ValidationEntry("source", "New warning.", "medium"),),
        )
        delta = _compute_validation_delta(before, after)
        # Error unchanged — no new, no resolved
        assert delta["new_errors"] == []
        assert delta["resolved_errors"] == []
        # Warning changed — old resolved, new appeared
        assert len(delta["new_warnings"]) == 1
        assert delta["new_warnings"][0]["message"] == "New warning."
        assert len(delta["resolved_warnings"]) == 1
        assert delta["resolved_warnings"][0]["message"] == "Old warning."

    def test_serialized_entries_include_severity(self) -> None:
        """Delta entries are serialized via to_dict() and include severity."""
        before = ValidationSummary(is_valid=True, errors=(), warnings=())
        after = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("node:a", "Broken.", "high"),),
        )
        delta = _compute_validation_delta(before, after)
        entry = delta["new_errors"][0]
        assert entry == {"component": "node:a", "message": "Broken.", "severity": "high"}


class TestInjectPriorValidation:
    """Tests for _inject_prior_validation — attaches pre-mutation validation."""

    def _make_result(
        self,
        *,
        success: bool,
        prior: ValidationSummary | None = None,
    ) -> ToolResult:
        state = _empty_state()
        return ToolResult(
            success=success,
            updated_state=state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=(),
            prior_validation=prior,
        )

    def test_injects_prior_on_success(self) -> None:
        """Successful mutation without prior_validation gets it injected."""
        prior = ValidationSummary(is_valid=False, errors=(ValidationEntry("source", "No source.", "high"),))
        result = self._make_result(success=True)
        assert result.prior_validation is None

        injected = _inject_prior_validation(result, prior)
        assert injected.prior_validation is prior
        assert injected.success is True

    def test_skips_injection_on_failure(self) -> None:
        """Failed mutation results are returned unchanged."""
        prior = ValidationSummary(is_valid=True, errors=())
        result = self._make_result(success=False)

        injected = _inject_prior_validation(result, prior)
        assert injected.prior_validation is None
        assert injected is result  # identity — unchanged

    def test_skips_injection_when_already_set(self) -> None:
        """Results that already carry prior_validation are not overwritten."""
        original_prior = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("node:a", "Handler set this.", "high"),),
        )
        new_prior = ValidationSummary(is_valid=True, errors=())
        result = self._make_result(success=True, prior=original_prior)

        injected = _inject_prior_validation(result, new_prior)
        assert injected.prior_validation is original_prior  # not overwritten
        assert injected is result  # identity — unchanged

    def test_to_dict_includes_delta_when_prior_set(self) -> None:
        """ToolResult.to_dict() includes validation_delta when prior_validation present."""
        prior = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("source", "No source.", "high"),),
        )
        state = _empty_state()
        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=(),
            prior_validation=prior,
        )
        d = result.to_dict()
        assert "validation_delta" in d
        assert len(d["validation_delta"]["resolved_errors"]) == 1
        assert d["validation_delta"]["new_errors"] == []

    def test_to_dict_omits_delta_when_no_prior(self) -> None:
        """ToolResult.to_dict() excludes validation_delta when no prior_validation."""
        result = self._make_result(success=True)
        d = result.to_dict()
        assert "validation_delta" not in d


class TestExecuteToolPriorValidation:
    """Integration: execute_tool populates prior_validation for mutation tools."""

    def test_mutation_tool_gets_prior_validation(self) -> None:
        """set_source (a mutation tool) should populate prior_validation."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert result.prior_validation is not None
        # Prior should reflect the original empty state's validation
        d = result.to_dict()
        assert "validation_delta" in d

    def test_discovery_tool_has_no_prior_validation(self) -> None:
        """list_sources (a discovery tool) should NOT have prior_validation."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("list_sources", {}, state, catalog)
        assert result.success is True
        assert result.prior_validation is None
        d = result.to_dict()
        assert "validation_delta" not in d

    def test_threaded_prior_used_for_mutation(self) -> None:
        """When prior_validation is threaded, execute_tool uses it as-is."""
        state = _empty_state()
        catalog = _mock_catalog()
        # Pre-compute validation for the empty state
        threaded = state.validate()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            prior_validation=threaded,
        )
        assert result.success is True
        # The threaded validation should be used as the prior — identity check
        assert result.prior_validation is threaded

    def test_threaded_prior_produces_correct_delta(self) -> None:
        """Threaded validation produces the same delta as fresh computation."""
        state = _empty_state()
        catalog = _mock_catalog()
        source_args = {
            "plugin": "csv",
            "on_success": "t1",
            "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
            "on_validation_failure": "quarantine",
        }
        # Without threading (fresh computation)
        result_fresh = execute_tool("set_source", source_args, state, catalog)
        # With threading
        threaded = state.validate()
        result_threaded = execute_tool(
            "set_source",
            source_args,
            state,
            catalog,
            prior_validation=threaded,
        )
        # Deltas should be identical
        delta_fresh = result_fresh.to_dict()["validation_delta"]
        delta_threaded = result_threaded.to_dict()["validation_delta"]
        assert delta_fresh == delta_threaded

    def test_chained_threading_across_mutations(self) -> None:
        """Validation chains correctly across sequential mutations."""
        state = _empty_state()
        catalog = _mock_catalog()

        # First mutation — no prior to thread
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "main",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert r1.success is True

        # Second mutation — thread r1's validation as prior
        r2 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r1.updated_state,
            catalog,
            prior_validation=r1.validation,
        )
        assert r2.success is True
        # r1.validation becomes r2's prior — identity check
        assert r2.prior_validation is r1.validation
        # The delta should reflect changes from r1's state to r2's state
        assert "validation_delta" in r2.to_dict()


class TestToolRegistry:
    """Tests for the tool registry pattern — two dicts + cacheable frozenset."""

    def test_discovery_tools_membership(self) -> None:
        """``_DISCOVERY_TOOLS`` keys equal the DISCOVERY-kind declaration name set.

        Step 6 falsification recovery (elspeth-6c9972ccbf, surface 1 of 2):
        the literal name enumeration previously here was a per-tool growth
        surface — every new discovery tool required updating this test in
        addition to its declaration. The literal was vestigial: it had no
        docstring framing it as a deliberate design-review gate and only
        duplicated the derivation pipeline that ``_DISCOVERY_TOOLS`` already
        runs at import time in ``_registry.py``.

        Retargeted at ``derive_name_set_for(_REGISTERED_TOOLS,
        ToolKind.DISCOVERY)`` so the assertion sources truth from the
        declaration tuple. The test still smoke-checks that
        ``_dispatch.py``'s consumed handler-map agrees with ``_registry.py``'s
        derivation, but no longer re-enumerates the names. The
        design-review gate for stateful discovery tools (the three
        ``diff_pipeline`` / ``get_pipeline_state`` / ``preview_pipeline``
        names) lives unchanged in
        ``test_cacheable_discovery_is_opt_in_with_named_mutable_complement``
        below, which retains its literal enumeration deliberately.
        """
        from elspeth.web.composer.tools import _DISCOVERY_TOOLS
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS
        from elspeth.web.composer.tools.declarations import (
            ToolKind,
            derive_name_set_for,
        )

        expected = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.DISCOVERY)
        assert set(_DISCOVERY_TOOLS.keys()) == expected

    def test_mutation_tools_membership(self) -> None:
        """``_MUTATION_TOOLS`` keys equal the MUTATION-kind declaration name set.

        See ``test_discovery_tools_membership`` for the rationale behind
        retargeting away from a literal enumeration. The plain-mutation kind
        has no design-review gate — every new mutation tool is just another
        declaration, and the literal here served only as accidental
        documentation that grew with every addition.
        """
        from elspeth.web.composer.tools import _MUTATION_TOOLS
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS
        from elspeth.web.composer.tools.declarations import (
            ToolKind,
            derive_name_set_for,
        )

        expected = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.MUTATION)
        assert set(_MUTATION_TOOLS.keys()) == expected

    def test_no_overlap_between_registries(self) -> None:
        from elspeth.web.composer.tools import _DISCOVERY_TOOLS, _MUTATION_TOOLS

        overlap = set(_DISCOVERY_TOOLS.keys()) & set(_MUTATION_TOOLS.keys())
        assert overlap == set(), f"Registry overlap: {overlap}"

    def test_cacheable_discovery_is_opt_in_with_named_mutable_complement(self) -> None:
        """Pin the *property* of the opt-in design, not the *result* of the
        subtraction the production code abandoned.

        Three load-bearing invariants the discovery module enforces at
        import time (``elspeth/web/composer/tools/discovery.py``):

        1. ``_SESSION_MUTABLE_DISCOVERY_TOOL_NAMES`` is a named, documented
           constant — surfacing the forbidden set as data (not a comment)
           so a future copy-paste edit can be mechanically rejected by the
           import-time assertion rather than silently auto-caching a new
           stateful discovery tool.
        2. The cacheable opt-in set and the mutable forbidden set are
           disjoint — the assertion at ``discovery.py:168-171`` would
           crash import time on a violation, this test additionally pins
           the runtime contract at the test layer.
        3. The contents of the mutable set name the three stateful
           discovery tools (``diff_pipeline``, ``get_pipeline_state``,
           ``preview_pipeline``) explicitly — adding a fourth requires
           updating this test, forcing a design-review checkpoint rather
           than letting the new tool slip in via subtraction arithmetic.

        Replaces the prior subtraction-shape assertion which mirrored
        the opt-OUT pattern the production code deliberately moved away
        from in commit e34f53c30.
        """
        from elspeth.web.composer.tools._registry import (
            _CACHEABLE_DISCOVERY_TOOL_NAMES,
            _DISCOVERY_TOOLS,
            _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES,
        )

        # Invariant 1: the forbidden set exists as named data.
        assert isinstance(_SESSION_MUTABLE_DISCOVERY_TOOL_NAMES, frozenset)

        # Invariant 2: disjointness — a tool cannot be both cacheable and
        # session-mutable. Belt-and-braces with the import-time assert.
        assert not (_CACHEABLE_DISCOVERY_TOOL_NAMES & _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES)

        # Invariant 3: the forbidden set names exactly the three stateful
        # tools. A fourth requires updating this test deliberately.
        assert frozenset({"diff_pipeline", "get_pipeline_state", "preview_pipeline"}) == _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES

        # Cross-check: every discovery tool is either cacheable or in the
        # documented forbidden set — no tool may live in neither category
        # by default (the opt-in regime requires an explicit classification
        # decision per tool).
        assert frozenset(_DISCOVERY_TOOLS.keys()) == (_CACHEABLE_DISCOVERY_TOOL_NAMES | _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES)

    def test_cacheable_is_subset_of_discovery(self) -> None:
        from elspeth.web.composer.tools import (
            _CACHEABLE_DISCOVERY_TOOL_NAMES,
            _DISCOVERY_TOOLS,
        )

        assert set(_DISCOVERY_TOOLS.keys()) >= _CACHEABLE_DISCOVERY_TOOL_NAMES

    def test_is_discovery_tool(self) -> None:
        from elspeth.web.composer.tools import is_discovery_tool

        assert is_discovery_tool("list_sources") is True
        assert is_discovery_tool("get_expression_grammar") is True
        assert is_discovery_tool("set_source") is False
        assert is_discovery_tool("nonexistent") is False

    def test_is_cacheable_discovery_tool(self) -> None:
        from elspeth.web.composer.tools import is_cacheable_discovery_tool

        assert is_cacheable_discovery_tool("list_sources") is True
        assert is_cacheable_discovery_tool("get_plugin_schema") is True
        assert is_cacheable_discovery_tool("set_source") is False

    def test_registry_dispatch_matches_original_behaviour(self) -> None:
        """Every tool in the registries dispatches correctly via execute_tool."""
        state = _empty_state()
        catalog = _mock_catalog()

        # All discovery tools should succeed
        for tool_name in [
            "list_sources",
            "list_transforms",
            "list_sinks",
            "get_expression_grammar",
        ]:
            result = execute_tool(tool_name, {}, state, catalog)
            assert result.success is True, f"{tool_name} failed"

        # get_plugin_schema needs arguments
        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "csv"},
            state,
            catalog,
        )
        assert result.success is True

        # Mutation tools that work on empty state
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert result.success is True

        result = execute_tool(
            "set_metadata",
            {"patch": {"name": "Test"}},
            state,
            catalog,
        )
        assert result.success is True

        # Unknown tool returns failure
        result = execute_tool("nonexistent", {}, state, catalog)
        assert result.success is False

    def test_module_level_assertion_no_overlap(self) -> None:
        """Importing the module should not raise — the overlap assertion passes."""
        import importlib

        import elspeth.web.composer.tools as mod

        importlib.reload(mod)  # Force re-evaluation of module-level assertion


# ---------------------------------------------------------------------------
# get_pipeline_state functional tests
# ---------------------------------------------------------------------------


class TestGetPipelineState:
    """Functional tests for get_pipeline_state — exercises all three modes
    (full state, component-specific, not-found) plus deep_thaw and redaction.
    """

    def _build_populated_state(self) -> CompositionState:
        """Build a state with source, node, output, and edge via tool calls."""
        state = _empty_state()
        catalog = _mock_catalog()

        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert r1.success is True

        r2 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "source",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
            },
            r1.updated_state,
            catalog,
        )
        assert r2.success is True

        r3 = execute_tool(
            "set_output",
            {
                "sink_name": "out",
                "plugin": "csv",
                "options": {"path": "/data/outputs/result.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r2.updated_state,
            catalog,
        )
        assert r3.success is True

        r4 = execute_tool(
            "upsert_edge",
            {"id": "e1", "from_node": "source", "to_node": "t1", "edge_type": "on_success"},
            r3.updated_state,
            catalog,
        )
        assert r4.success is True

        return r4.updated_state

    def test_full_state_returns_all_components(self) -> None:
        """No component arg returns source, nodes, outputs, edges, metadata."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {}, state, catalog)
        assert result.success is True

        # Use to_dict() for structural checks — result.data is frozen by ToolResult.__post_init__
        data = result.to_dict()["data"]
        assert _pipeline_state_default_source(data) is not None
        assert _pipeline_state_default_source(data)["plugin"] == "csv"
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == "t1"
        assert len(data["outputs"]) == 1
        assert data["outputs"][0]["sink_name"] == "out"
        assert len(data["edges"]) == 1
        assert data["edges"][0]["id"] == "e1"
        assert "metadata" in data
        assert "version" in data

    def test_full_state_alias_full_returns_all_components(self) -> None:
        """component='full' is accepted as an explicit full-state alias."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": "full"}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert data["inspection"]["resolved_component"] == "full"
        assert "full" in data["inspection"]["accepted_full_state_aliases"]
        assert _pipeline_state_default_source(data) is not None
        assert data["nodes"][0]["id"] == "t1"
        assert data["outputs"][0]["sink_name"] == "out"
        assert data["edges"][0]["id"] == "e1"

    def test_full_state_alias_empty_string_returns_all_components(self) -> None:
        """component='' is accepted as an explicit full-state alias."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": ""}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert data["inspection"]["requested_component"] == ""
        assert data["inspection"]["resolved_component"] == "full"
        assert data["nodes"][0]["id"] == "t1"
        assert data["outputs"][0]["sink_name"] == "out"

    def test_full_state_options_are_plain_dicts(self) -> None:
        """to_dict() deep_thaw converts frozen containers to plain dicts for JSON serialization."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {}, state, catalog)
        assert result.success is True

        # to_dict() runs deep_thaw on result.data — options must be plain dicts
        data = result.to_dict()["data"]
        source_opts = _pipeline_state_default_source(data)["options"]
        assert isinstance(source_opts, dict)
        assert isinstance(source_opts.get("schema"), dict)

        node_opts = data["nodes"][0]["options"]
        assert isinstance(node_opts, dict)

    def test_component_source(self) -> None:
        """component='source' returns only the source component."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": "source"}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert "sources" in data
        assert _pipeline_state_default_source(data)["plugin"] == "csv"
        # Should not contain nodes/outputs/edges
        assert "nodes" not in data
        assert "outputs" not in data

    def test_component_source_when_none(self) -> None:
        """component='source' with no source set returns null source."""
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": "source"}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert _pipeline_state_default_source(data) is None

    def test_component_node_by_id(self) -> None:
        """component=<node_id> returns that node's details."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": "t1"}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert "node" in data
        assert data["node"]["id"] == "t1"
        assert data["node"]["plugin"] == "passthrough"
        assert isinstance(data["node"]["options"], dict)

    def test_full_state_alias_does_not_shadow_real_node_id(self) -> None:
        """A real node id named 'full' remains addressable as a component."""
        state = _empty_state().with_node(
            NodeSpec(
                id="full",
                node_type="transform",
                plugin="passthrough",
                input="source",
                on_success="out",
                on_error=None,
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": "full"}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert "node" in data
        assert data["node"]["id"] == "full"
        assert "nodes" not in data

    def test_component_output_by_name(self) -> None:
        """component=<output_name> returns that output's details."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": "out"}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert "output" in data
        assert data["output"]["sink_name"] == "out"
        assert data["output"]["plugin"] == "csv"

    def test_component_not_found(self) -> None:
        """component=<nonexistent> returns failure."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {"component": "nonexistent"}, state, catalog)
        assert result.success is False

    def test_empty_state_full_returns_nulls(self) -> None:
        """Full state on empty pipeline returns null source and empty lists."""
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        assert _pipeline_state_default_source(data) is None
        assert data["nodes"] == []
        assert data["outputs"] == []
        assert data["edges"] == []

    def test_no_prior_validation(self) -> None:
        """get_pipeline_state is a discovery tool — no prior_validation."""
        state = self._build_populated_state()
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {}, state, catalog)
        assert result.prior_validation is None

    def test_blob_ref_source_path_redacted(self) -> None:
        """When source has blob_ref, internal storage path is redacted (B4)."""
        source = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"path": "/internal/blobs/abc123.csv", "blob_ref": "abc123", "schema": {"mode": "observed"}},
            on_validation_failure="quarantine",
        )
        state = CompositionState(
            source=source,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        catalog = _mock_catalog()

        result = execute_tool("get_pipeline_state", {}, state, catalog)
        assert result.success is True
        data = result.to_dict()["data"]
        # path key remains visible so the LLM can tell the source is configured,
        # but the internal storage value itself is not exposed.
        assert _pipeline_state_default_source(data)["options"]["path"] == EXPECTED_REDACTED_BLOB_SOURCE_PATH
        assert _pipeline_state_default_source(data)["options"]["blob_ref"] == "abc123"
        assert "/internal/blobs/abc123.csv" not in str(data)


# ---------------------------------------------------------------------------
# Blob tool tests — composer-level security boundaries
# ---------------------------------------------------------------------------


class TestBlobTools:
    """Blob composition tools: session context enforcement, storage_path exclusion,
    status guards, and source plugin wiring.

    Security contracts tested:
    - Blob tools fail without session context (no ambient access)
    - get_blob_metadata never exposes storage_path to the LLM
    - Wrong session_id returns failure (IDOR at the tool layer)
    - set_source_from_blob rejects non-ready blobs
    - set_source_from_blob wires the correct source plugin from MIME type
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Create an in-memory SQLite engine with tables and seed data."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.other_session_id = str(uuid4())
        self.blob_id = str(uuid4())
        self.pending_blob_id = str(uuid4())

        now = datetime.now(UTC)

        with self.engine.begin() as conn:
            # Two sessions
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test Session",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                sessions_table.insert().values(
                    id=self.other_session_id,
                    user_id="other-user",
                    auth_provider_type="local",
                    title="Other Session",
                    created_at=now,
                    updated_at=now,
                )
            )
            # Ready blob in session
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=100,
                    content_hash=_STUB_SHA256,
                    storage_path="/tmp/fake/data.csv",
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )
            # Pending blob in session
            conn.execute(
                blobs_table.insert().values(
                    id=self.pending_blob_id,
                    session_id=self.session_id,
                    filename="output.csv",
                    mime_type="text/csv",
                    size_bytes=0,
                    content_hash=None,
                    storage_path="/tmp/fake/output.csv",
                    created_at=now,
                    created_by="pipeline",
                    source_description=None,
                    status="pending",
                )
            )

    def test_list_blobs_without_session_context_returns_failure(self) -> None:
        """Blob tools with no session context must fail — no ambient data access."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("list_blobs", {}, state, catalog)
        assert result.success is False

    def test_get_blob_metadata_excludes_storage_path(self) -> None:
        """storage_path must never be exposed to the LLM — it leaks filesystem layout."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_blob_metadata",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )
        assert result.success is True
        assert "storage_path" not in result.data

    def test_get_blob_metadata_wrong_session_returns_failure(self) -> None:
        """IDOR at tool layer: blob belongs to session A, caller is session B."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_blob_metadata",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.other_session_id,
        )
        assert result.success is False

    def _tamper_blob_row(self, **values: Any) -> None:
        """Bypass CHECK constraints so defensive read guards can be tested."""
        from sqlalchemy import text

        from elspeth.web.sessions.models import blobs_table

        with self.engine.begin() as conn:
            conn.execute(text("PRAGMA ignore_check_constraints = 1"))
            conn.execute(blobs_table.update().where(blobs_table.c.id == self.blob_id).values(**values))
            conn.execute(text("PRAGMA ignore_check_constraints = 0"))

    def test_get_blob_metadata_tampered_status_raises_audit_integrity_error(self) -> None:
        """Corrupted blob status must crash instead of leaking to the LLM."""
        from elspeth.contracts.errors import AuditIntegrityError

        self._tamper_blob_row(status="corrupted")

        with pytest.raises(AuditIntegrityError, match=r"blobs\.status"):
            execute_tool(
                "get_blob_metadata",
                {"blob_id": self.blob_id},
                _empty_state(),
                _mock_catalog(),
                session_engine=self.engine,
                session_id=self.session_id,
            )

    def test_list_blobs_tampered_mime_type_raises_audit_integrity_error(self) -> None:
        """Corrupted MIME allowlist values must crash instead of being listed."""
        from elspeth.contracts.errors import AuditIntegrityError

        self._tamper_blob_row(mime_type="TEXT/CSV")

        with pytest.raises(AuditIntegrityError, match=r"blobs\.mime_type"):
            execute_tool(
                "list_blobs",
                {},
                _empty_state(),
                _mock_catalog(),
                session_engine=self.engine,
                session_id=self.session_id,
            )

    def test_set_source_from_blob_rejects_non_ready(self) -> None:
        """Cannot wire a pending blob as source — content doesn't exist yet."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": self.pending_blob_id, "on_success": "out"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )
        assert result.success is False

    def test_set_source_from_blob_wires_correct_plugin(self) -> None:
        """text/csv blob should auto-resolve to the 'csv' source plugin."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": self.blob_id, "on_success": "out", "options": {"schema": {"mode": "observed"}}},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "post,run\n1,1"),
        )
        assert result.success is True
        assert _default_source(result.updated_state) is not None
        assert _default_source(result.updated_state).plugin == "csv"
        assert _default_source(result.updated_state).on_validation_failure == "discard"

    def test_set_source_from_blob_wires_named_source(self) -> None:
        """Blob-backed sources must be reachable as explicit named roots."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source_from_blob",
            {
                "blob_id": self.blob_id,
                "source_name": "orders",
                "on_success": "orders_rows",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )

        assert result.success is True
        assert result.updated_state.sources["orders"].plugin == "csv"
        assert result.updated_state.sources["orders"].options["blob_ref"] == self.blob_id
        assert result.affected_nodes == ("source:orders",)

    @pytest.mark.parametrize(
        "source_name",
        ("Orders", "on_success", " "),
    )
    def test_set_source_from_blob_invalid_source_name_raises_arg_error(self, source_name: str) -> None:
        """Invalid blob source names are LLM argument errors, not plugin crashes."""
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()

        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "set_source_from_blob",
                {
                    "blob_id": self.blob_id,
                    "source_name": source_name,
                    "on_success": "orders_rows",
                    "options": {"schema": {"mode": "observed"}},
                },
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
            )

        assert exc_info.value.argument == "source_name"
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_set_source_from_plain_text_blob_uses_text_source(self) -> None:
        """text/plain blob should auto-resolve to the 'text' source plugin."""
        from elspeth.web.sessions.models import blobs_table

        state = _empty_state()
        catalog = _mock_catalog()

        with self.engine.begin() as conn:
            conn.execute(blobs_table.update().where(blobs_table.c.id == self.blob_id).values(mime_type="text/plain"))

        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": self.blob_id, "on_success": "out", "options": {"schema": {"mode": "observed"}, "column": "line"}},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )

        assert result.success is True
        assert _default_source(result.updated_state) is not None
        assert _default_source(result.updated_state).plugin == "text"

    def test_set_source_from_jsonl_blob_uses_json_plugin_with_format(self) -> None:
        """Regression: JSONL MIME types must resolve to 'json' plugin with format='jsonl'."""
        from elspeth.web.sessions.models import blobs_table

        state = _empty_state()
        catalog = _mock_catalog()

        with self.engine.begin() as conn:
            conn.execute(blobs_table.update().where(blobs_table.c.id == self.blob_id).values(mime_type="application/x-jsonlines"))

        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": self.blob_id, "on_success": "out", "options": {"schema": {"mode": "observed"}}},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )

        assert result.success is True
        assert _default_source(result.updated_state) is not None
        assert _default_source(result.updated_state).plugin == "json"
        assert _default_source(result.updated_state).options["format"] == "jsonl"

    def test_set_source_from_blob_merges_caller_options(self) -> None:
        """Caller-provided options are merged with blob-derived options.

        Plugin-specific config like schema and column must flow through,
        while path and blob_ref remain authoritative from the blob.
        """
        from elspeth.web.sessions.models import blobs_table

        state = _empty_state()
        catalog = _mock_catalog()

        # Update the test blob to text/plain so we get the text plugin
        with self.engine.begin() as conn:
            conn.execute(blobs_table.update().where(blobs_table.c.id == self.blob_id).values(mime_type="text/plain"))

        result = execute_tool(
            "set_source_from_blob",
            {
                "blob_id": self.blob_id,
                "on_success": "out",
                "options": {
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
            },
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "should,proceed\n1,1"),
        )

        assert result.success is True
        assert _default_source(result.updated_state) is not None
        assert _default_source(result.updated_state).plugin == "text"
        # Caller options merged in
        assert _default_source(result.updated_state).options["column"] == "line"
        assert _default_source(result.updated_state).options["schema"] == {"mode": "observed"}
        # Blob-derived options still present (path is internal, blob_ref is visible)
        assert "blob_ref" in _default_source(result.updated_state).options
        assert _default_source(result.updated_state).options["blob_ref"] == self.blob_id

    def test_set_source_from_blob_blob_options_override_caller(self) -> None:
        """Blob-derived path and blob_ref cannot be overridden by caller.

        This is a security constraint: the blob's storage path is authoritative.
        Callers cannot inject an arbitrary path via the options parameter.
        """
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool(
            "set_source_from_blob",
            {
                "blob_id": self.blob_id,
                "on_success": "out",
                "options": {
                    "path": "/etc/passwd",  # Attempted path injection
                    "blob_ref": "malicious-ref",
                    "schema": {"mode": "observed"},
                },
            },
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "would,corrupt,mid-run\n"),
        )

        assert result.success is True
        assert _default_source(result.updated_state) is not None
        # Blob's path and ref take precedence — caller cannot override
        assert _default_source(result.updated_state).options["blob_ref"] == self.blob_id
        assert _default_source(result.updated_state).options["path"] != "/etc/passwd"

    def test_set_source_from_blob_gets_prior_validation(self) -> None:
        """Blob mutation tools must populate prior_validation for validation delta."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": self.blob_id, "on_success": "out", "options": {"schema": {"mode": "observed"}}},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert result.prior_validation is not None
        d = result.to_dict()
        assert "validation_delta" in d

    def test_set_source_from_blob_threaded_prior_used(self) -> None:
        """Threaded prior_validation is reused by blob mutation tools (identity check).

        execute_tool dispatches blob mutations through a separate branch from
        standard mutations. Both branches must honour the prior_validation kwarg.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        threaded = state.validate()
        result = execute_tool(
            "set_source_from_blob",
            {"blob_id": self.blob_id, "on_success": "out", "options": {"schema": {"mode": "observed"}}},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            prior_validation=threaded,
        )
        assert result.success is True
        assert result.prior_validation is threaded

    def test_set_source_from_blob_threaded_prior_produces_correct_delta(self) -> None:
        """Threaded and fresh prior_validation produce identical deltas for blob tools."""
        state = _empty_state()
        catalog = _mock_catalog()
        blob_args: dict[str, Any] = {
            "blob_id": self.blob_id,
            "on_success": "out",
            "options": {"schema": {"mode": "observed"}},
        }
        # Fresh (no threading)
        result_fresh = execute_tool(
            "set_source_from_blob",
            blob_args,
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        # Threaded
        threaded = state.validate()
        result_threaded = execute_tool(
            "set_source_from_blob",
            blob_args,
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            prior_validation=threaded,
        )
        delta_fresh = result_fresh.to_dict()["validation_delta"]
        delta_threaded = result_threaded.to_dict()["validation_delta"]
        assert delta_fresh == delta_threaded

    def test_set_source_from_blob_unknown_vf_sink_includes_note(self) -> None:
        """Blob-backed source with unknown on_validation_failure gets advisory note."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source_from_blob",
            {
                "blob_id": self.blob_id,
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
                "on_validation_failure": "nonexistent",
            },
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert result.data is not None
        assert "nonexistent" in result.data["note"]
        assert "discard" in result.data["note"]

    def test_create_blob_cleans_file_on_db_failure(self, tmp_path: Path) -> None:
        """DB failure during create_blob must delete the orphaned storage file."""
        from unittest.mock import patch

        state = _empty_state()
        catalog = _mock_catalog()
        data_dir = str(tmp_path)

        # Patch _check_blob_quota to raise inside the DB transaction
        with (
            patch(
                "elspeth.web.composer.tools.blobs._check_blob_quota",
                side_effect=RuntimeError("simulated DB failure"),
            ),
            pytest.raises(RuntimeError, match="simulated DB failure"),
        ):
            execute_tool(
                "create_blob",
                {"filename": "test.csv", "mime_type": "text/csv", "content": "a,b\n1,2"},
                state,
                catalog,
                data_dir=data_dir,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "a,b\n1,2"),
            )

        # Storage file must have been cleaned up
        blob_dir = tmp_path / "blobs" / self.session_id
        remaining = list(blob_dir.glob("*")) if blob_dir.exists() else []
        assert remaining == [], f"Orphaned files after DB failure: {remaining}"

    def test_update_blob_restores_old_content_on_db_failure(self, tmp_path: Path) -> None:
        """DB failure during update_blob must restore the original file content."""
        from datetime import UTC, datetime
        from unittest.mock import patch
        from uuid import uuid4

        from elspeth.web.sessions.models import blobs_table

        state = _empty_state()
        catalog = _mock_catalog()

        # Create a real blob on disk with known content
        blob_id = str(uuid4())
        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        storage_path = storage_dir / f"{blob_id}_test.csv"
        original_content = b"original,content\n1,2"
        storage_path.write_bytes(original_content)

        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=blob_id,
                    session_id=self.session_id,
                    filename="test.csv",
                    mime_type="text/csv",
                    size_bytes=len(original_content),
                    content_hash=_STUB_SHA256,
                    storage_path=str(storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

        # Patch session_engine.begin() to raise AFTER the file is overwritten.
        # The update function reads old content, writes new content, THEN enters
        # the DB transaction.  We need the DB part to fail.
        provenance_context = _verbatim_blob_context(self.engine, self.session_id, "new,content\n3,4")
        with (
            patch.object(
                self.engine,
                "begin",
                side_effect=RuntimeError("simulated DB failure"),
            ),
            pytest.raises(RuntimeError, match="simulated DB failure"),
        ):
            execute_tool(
                "update_blob",
                {"blob_id": blob_id, "content": "new,content\n3,4"},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **provenance_context,
            )

        # File must contain the ORIGINAL content after rollback
        assert storage_path.read_bytes() == original_content

    def test_blob_rollback_does_not_catch_keyboard_interrupt(self, tmp_path: Path) -> None:
        """Blob exception handlers must catch Exception, not BaseException.

        Catching BaseException intercepts KeyboardInterrupt/SystemExit.
        Under KeyboardInterrupt, write_bytes() rollback (update_blob) could
        truncate the file, leaving it inconsistent with DB state.
        """
        import ast
        import inspect

        from elspeth.web.composer.tools import _execute_create_blob, _execute_update_blob

        for func in (_execute_create_blob, _execute_update_blob):
            source = inspect.getsource(func)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is not None and isinstance(node.type, ast.Name):
                    assert node.type.id != "BaseException", (
                        f"{func.__name__} catches BaseException — must use Exception to avoid intercepting KeyboardInterrupt/SystemExit"
                    )


# ---------------------------------------------------------------------------
# Blob active-run protection (Finding 2: 73a1aa6cef)
# ---------------------------------------------------------------------------


class TestDeleteBlobActiveRunGuard:
    """delete_blob must refuse to delete blobs linked to active (pending/running) runs.

    Mirrors BlobServiceImpl.delete_blob() active-run guard — the composer tool
    layer must enforce the same invariant.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        self.run_id = str(uuid4())
        now = datetime.now(UTC)

        # Create blob on disk so unlink has a real target
        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_data.csv"
        self.storage_path.write_bytes(b"a,b\n1,2")

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=100,
                    content_hash=_STUB_SHA256,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def _insert_run_and_link(self, status: str) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from elspeth.web.sessions.models import (
            blob_run_links_table,
            composition_states_table,
            runs_table,
        )

        now = datetime.now(UTC)
        state_id = str(uuid4())
        with self.engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=self.session_id,
                    version=1,
                    source=None,
                    nodes=None,
                    edges=None,
                    outputs=None,
                    metadata_=None,
                    is_valid=False,
                    validation_errors=None,
                    # Plan §2294: composer-tools test fixture; provenance
                    # required for setup row supporting subsequent runs/
                    # blob_run_links FKs.
                    provenance="session_seed",
                    created_at=now,
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=self.run_id,
                    session_id=self.session_id,
                    state_id=state_id,
                    status=status,
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                )
            )
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=self.blob_id,
                    run_id=self.run_id,
                    direction="input",
                )
            )

    def _insert_run_without_link(self, status: str, *, source: dict[str, Any] | None = None) -> None:
        """Insert a run in the blob's session but omit the blob_run_links row.

        Simulates the pre-link window: _execute_locked() has called
        create_run() but link_blob_to_run() hasn't fired yet.

        Args:
            source: Composition state source dict.  Defaults to a source
                that references self.blob_id via blob_ref (the typical
                pre-link scenario).  Pass a different dict to simulate
                runs that use file-path sources with no blob_ref.
        """
        from datetime import UTC, datetime
        from uuid import uuid4

        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        if source is None:
            source = {
                "plugin": "csv",
                "on_success": "output",
                "on_validation_failure": "quarantine",
                "options": {"blob_ref": self.blob_id, "path": str(self.storage_path)},
            }
        else:
            source = {
                "on_success": "output",
                "on_validation_failure": "quarantine",
                **source,
            }

        now = datetime.now(UTC)
        state_id = str(uuid4())
        with self.engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=self.session_id,
                    version=1,
                    source=source,
                    nodes=[],
                    edges=[],
                    outputs=[],
                    metadata_={"name": "Test", "description": ""},
                    is_valid=False,
                    validation_errors=None,
                    # Plan §2294: composer-tools test fixture; provenance
                    # required for setup row supporting subsequent runs/
                    # blob_run_links FKs.
                    provenance="session_seed",
                    created_at=now,
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=self.run_id,
                    session_id=self.session_id,
                    state_id=state_id,
                    status=status,
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                )
            )

    def test_delete_succeeds_when_no_runs_linked(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert not self.storage_path.exists()

    def test_delete_rejected_when_pending_run_exists_without_link(self) -> None:
        """Pre-link window: run exists but blob_run_links row hasn't been created yet.

        _execute_locked() creates the run record before link_blob_to_run() inserts
        the link row.  During that gap, the explicit-link guard sees nothing.
        The composition-state guard must block deletion because the run's source
        references this blob via blob_ref.
        """
        self._insert_run_without_link("pending")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.exists(), "File must not be deleted when guard blocks"

    def test_delete_rejected_when_running_run_exists_without_link(self) -> None:
        """Same as pending — a running run without a link row must also block."""
        self._insert_run_without_link("running")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.exists(), "File must not be deleted when guard blocks"

    def test_delete_succeeds_when_active_run_uses_different_source(self) -> None:
        """Active run using source.path (no blob_ref) must not block unrelated blob deletion.

        Regression test: the original session-level guard blocked ALL blobs
        when ANY run was active, even if that run used a file-path source.
        The scoped guard checks source.options.blob_ref and only blocks
        if it matches this blob.
        """
        self._insert_run_without_link(
            "pending",
            source={"plugin": "csv", "options": {"path": "/data/external/other.csv"}},
        )
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert not self.storage_path.exists()

    def test_delete_rejected_when_active_run_path_matches_storage(self) -> None:
        """Active run using source.path matching this blob's storage_path must block.

        A run can read a blob's backing file via plain set_source with
        options.path (no blob_ref).  The guard must check path/file matches.
        """
        self._insert_run_without_link(
            "pending",
            source={"plugin": "csv", "options": {"path": str(self.storage_path)}},
        )
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.exists(), "File must not be deleted when guard blocks"

    def test_delete_succeeds_when_completed_run_exists_without_link(self) -> None:
        """Completed runs (no link row) must not block deletion."""
        self._insert_run_without_link("completed")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert not self.storage_path.exists()

    def test_delete_rejected_when_pending_run_linked(self) -> None:
        self._insert_run_and_link("pending")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.exists(), "File must not be deleted when guard blocks"

    def test_delete_rejected_when_running_run_linked(self) -> None:
        self._insert_run_and_link("running")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert self.storage_path.exists(), "File must not be deleted when guard blocks"

    def test_delete_succeeds_when_completed_run_linked(self) -> None:
        self._insert_run_and_link("completed")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "delete_blob",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert not self.storage_path.exists()

    def test_delete_restores_file_when_db_delete_fails_after_filesystem_mutation(self) -> None:
        """DB failure after the filesystem step must not leave a stale row/missing-file split."""
        from contextlib import contextmanager

        from sqlalchemy import select

        from elspeth.web.sessions.models import blobs_table

        real_begin = self.engine.begin

        @contextmanager
        def failing_begin():
            with real_begin() as real_conn:

                class Proxy:
                    def __getattr__(self, name: str) -> Any:
                        return getattr(real_conn, name)

                    def execute(self, stmt, *args, **kwargs):
                        if str(stmt).lstrip().upper().startswith("DELETE FROM BLOBS"):
                            raise RuntimeError("simulated delete failure")
                        return real_conn.execute(stmt, *args, **kwargs)

                yield Proxy()

        self.engine.begin = failing_begin  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="simulated delete failure"):
                execute_tool(
                    "delete_blob",
                    {"blob_id": self.blob_id},
                    _empty_state(),
                    _mock_catalog(),
                    session_engine=self.engine,
                    session_id=self.session_id,
                )
        finally:
            self.engine.begin = real_begin  # type: ignore[method-assign]

        assert self.storage_path.exists(), "Rollback must restore the backing file"
        with self.engine.connect() as conn:
            row = conn.execute(select(blobs_table.c.id).where(blobs_table.c.id == self.blob_id)).first()
        assert row is not None, "The blob row should still exist after the failed delete"


# ---------------------------------------------------------------------------
# Blob update quota enforcement (Finding 5: 527546bedb)
# ---------------------------------------------------------------------------


class TestUpdateBlobQuota:
    """update_blob must enforce per-session quota when the blob grows.

    Mirrors _execute_create_blob quota enforcement — the update path must
    also call _check_blob_quota atomically inside the same transaction as
    the DB UPDATE.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        self.data_dir = str(tmp_path)
        now = datetime.now(UTC)

        # Create blob on disk with known content
        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_data.csv"
        self.original_content = b"small"
        self.storage_path.write_bytes(self.original_content)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.original_content),
                    content_hash=_STUB_SHA256,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def test_check_blob_quota_locks_session_before_sum(self, monkeypatch) -> None:
        """Composer blob writers use the same session row serialization guard."""
        from elspeth.web.composer.tools import blobs as composer_blob_tools

        locked_sessions: list[str] = []
        original_lock = composer_blob_tools._lock_session_for_blob_quota

        def recording_lock(conn, session_id: str) -> None:
            locked_sessions.append(session_id)
            original_lock(conn, session_id)

        monkeypatch.setattr(composer_blob_tools, "_lock_session_for_blob_quota", recording_lock)

        with self.engine.begin() as conn:
            quota_error = composer_blob_tools._check_blob_quota(
                conn,
                self.session_id,
                additional_bytes=1,
                quota_bytes=100,
            )

        assert quota_error is None
        assert locked_sessions == [self.session_id]

    def test_update_locks_session_before_current_size_delta_read(self, monkeypatch) -> None:
        """The quota lock must cover the current-size read used for update deltas."""
        from elspeth.web.composer.tools import blobs as composer_blob_tools

        events: list[str] = []
        original_lock = composer_blob_tools._lock_session_for_blob_quota

        def recording_lock(conn, session_id: str) -> None:
            events.append("lock")
            original_lock(conn, session_id)

        def record_size_read(_conn, _cursor, statement: str, _parameters, _context, _executemany) -> None:
            normalized = " ".join(statement.lower().split())
            if normalized.startswith("select blobs.size_bytes") and "from blobs" in normalized:
                events.append("size_read")

        monkeypatch.setattr(composer_blob_tools, "_lock_session_for_blob_quota", recording_lock)
        event.listen(self.engine, "before_cursor_execute", record_size_read)
        try:
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "larger content"},
                _empty_state(),
                _mock_catalog(),
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "larger content"),
            )
        finally:
            event.remove(self.engine, "before_cursor_execute", record_size_read)

        assert result.success is True
        assert events[:2] == ["lock", "size_read"]

    def test_update_within_quota_succeeds(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "slightly larger content"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "slightly larger content"),
        )
        assert result.success is True

    def test_update_exceeding_quota_rejected(self) -> None:
        from unittest.mock import patch

        state = _empty_state()
        catalog = _mock_catalog()
        # Set quota to a tiny value so any growth exceeds it
        with patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", 10):
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "x" * 100},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "x" * 100),
            )
        assert result.success is False
        assert "quota" in result.data["error"].lower()

    def test_update_exceeding_quota_preserves_old_content(self) -> None:
        from unittest.mock import patch

        state = _empty_state()
        catalog = _mock_catalog()
        with patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", 10):
            execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "x" * 100},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "x" * 100),
            )
        assert self.storage_path.read_bytes() == self.original_content

    def test_shrink_always_succeeds(self) -> None:
        from unittest.mock import patch

        # First grow the blob so we have something to shrink
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "a" * 200},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "a" * 200),
        )
        assert result.success is True

        # Now set quota very low — shrinking should still succeed
        with patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", 10):
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "tiny"},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "tiny"),
            )
        assert result.success is True

    def test_delta_boundary_case(self) -> None:
        """Update that fits when measured by delta but not by absolute new size.

        Session total = 490 (including current blob at old_size=5).
        New content = 15 bytes. Delta = 10. Total after = 490 + 10 = 500.
        Must succeed at quota=500 because 500 <= 500.
        Would fail if check incorrectly used full len(content_bytes)=15.
        """
        from datetime import UTC, datetime
        from unittest.mock import patch
        from uuid import uuid4

        from elspeth.web.sessions.models import blobs_table

        # Add a second blob to bring session total to 490
        filler_id = str(uuid4())
        now = datetime.now(UTC)
        filler_path = Path(self.data_dir) / "blobs" / self.session_id / f"{filler_id}_filler.bin"
        filler_path.write_bytes(b"x" * 485)

        with self.engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=filler_id,
                    session_id=self.session_id,
                    filename="filler.bin",
                    mime_type="application/octet-stream",
                    size_bytes=485,
                    content_hash=_STUB_SHA256_ALT,
                    storage_path=str(filler_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )
        # Session total is now 5 (original) + 485 (filler) = 490

        state = _empty_state()
        catalog = _mock_catalog()
        # New content: 15 bytes. Delta = 15 - 5 = 10. Total after = 490 + 10 = 500.
        with patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", 500):
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "x" * 15},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "x" * 15),
            )
        assert result.success is True, f"Delta-based quota check should pass at boundary: {result.data}"

    def test_shrink_on_at_quota_session_succeeds(self) -> None:
        """Shrinking a blob on a session exactly at quota must succeed."""
        from unittest.mock import patch

        state = _empty_state()
        catalog = _mock_catalog()
        # Quota exactly matches current total (5 bytes)
        with patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", len(self.original_content)):
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "x"},  # 1 byte < 5 bytes
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "x"),
            )
        assert result.success is True

    def test_quota_delta_uses_current_db_size_not_stale_snapshot(self) -> None:
        """size_delta must be computed from the in-transaction DB row, not the
        pre-transaction snapshot returned by _sync_get_blob().

        Scenario: blob starts at 5 bytes.  A concurrent writer grows it to 50
        bytes between the _sync_get_blob() call and the transaction.  Our
        update writes 60 bytes.  The correct delta is 60 - 50 = 10, not
        60 - 5 = 55.  With quota set to 70, the stale delta (55) would exceed
        quota while the correct delta (10) fits.
        """
        from unittest.mock import patch

        from sqlalchemy import update as sa_update

        # Simulate concurrent writer: bump DB size_bytes to 50 *after*
        # _sync_get_blob() has already read 5.  We hook _sync_get_blob to
        # perform the concurrent write immediately after returning.
        from elspeth.web.composer.tools.blobs import _sync_get_blob as original_get
        from elspeth.web.sessions.models import blobs_table

        def _get_then_concurrent_write(*args, **kwargs):
            result = original_get(*args, **kwargs)
            # Simulate concurrent writer updating size_bytes in the DB
            with self.engine.begin() as conn:
                conn.execute(sa_update(blobs_table).where(blobs_table.c.id == self.blob_id).values(size_bytes=50))
            return result

        state = _empty_state()
        catalog = _mock_catalog()

        # Quota = 70.  Correct delta: 60 - 50 = 10 → total 70 ≤ 70 → OK.
        # Stale delta: 60 - 5 = 55 → total 70 (from SUM) + 55 = would exceed.
        # (But _check_blob_quota reads SUM which already includes the 50,
        #  so stale delta of 55 → 50 + 55 = 105 > 70 → wrongly rejected.)
        with (
            patch("elspeth.web.composer.tools.blobs._sync_get_blob", side_effect=_get_then_concurrent_write),
            patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", 70),
        ):
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "x" * 60},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "x" * 60),
            )
        assert result.success is True, f"Quota check used stale snapshot instead of current DB size: {result.data}"


class TestUpdateBlobRollbackPreservesPrimaryException:
    """Pre-``os.replace`` failures must propagate cleanly with storage intact.

    Post atomic-rename refactor (bug_004), ``_execute_update_blob``
    writes new content to a sibling tempfile and defers the file swap
    to ``os.replace`` inside the DB transaction — AFTER the active-run
    guard, quota check, and UPDATE.  Any failure BEFORE ``os.replace``
    therefore cannot produce file/DB divergence because the backing
    file was never touched, and the rollback-write branch must be
    skipped (writing ``old_content`` back would be a needless write on
    an unmodified file).

    Before the refactor the file was overwritten BEFORE the DB
    transaction, so every DB failure required a rollback-write and an
    add_note-on-rollback-OSError discipline.  That discipline is
    retained in the code for the narrowed post-replace commit-failure
    window (still reachable via ``except Exception`` when ``replaced``
    is True), but the pre-replace scenarios — which dominate in
    practice — now exit cleanly.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        self.data_dir = str(tmp_path)
        now = datetime.now(UTC)

        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_data.csv"
        self.original_content = b"original-bytes"
        self.storage_path.write_bytes(self.original_content)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.original_content),
                    content_hash=_STUB_SHA256,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def test_primary_db_exception_before_replace_propagates_without_rollback(self) -> None:
        """An in-transaction failure before ``os.replace`` must not trigger a rollback write.

        Forces ``_check_blob_quota`` to raise RuntimeError mid-transaction.
        Because the atomic-rename flow defers the file swap to
        ``os.replace`` AFTER the quota check, storage_path is never
        modified — the RuntimeError propagates cleanly with:

        * no call to the rollback write branch (``replaced`` was never
          set to True);
        * no add_note diagnostic (no divergence occurred);
        * storage_path still containing the original bytes;
        * no stale tempfile in the storage directory.
        """
        from unittest.mock import patch

        from elspeth.web.composer.tools import _execute_update_blob

        primary_message = "primary-db-fault"

        def _raise_primary(*_args: Any, **_kwargs: Any) -> str | None:
            raise RuntimeError(primary_message)

        # A failing rollback-write patch WOULD be armed in the
        # pre-fix design; under the new design no rollback-write
        # runs so the patch is a negative guard — if any write to
        # storage_path happens after the first read, the test fails
        # via the tripwire counter.
        target_path_str = str(self.storage_path)
        write_bytes_calls_to_storage = [0]
        real_write_bytes = Path.write_bytes

        def _tripwire_write_bytes(path_self: Path, data: bytes) -> int:
            if str(path_self) == target_path_str:
                write_bytes_calls_to_storage[0] += 1
            return real_write_bytes(path_self, data)

        state = _empty_state()
        catalog = _mock_catalog()

        with (
            patch("elspeth.web.composer.tools.blobs._check_blob_quota", side_effect=_raise_primary),
            patch.object(Path, "write_bytes", _tripwire_write_bytes),
            pytest.raises(RuntimeError, match=primary_message) as exc_info,
        ):
            from elspeth.web.composer.tools._common import ToolContext as _UpdateBlobCtx

            _execute_update_blob(
                {"blob_id": self.blob_id, "content": "x" * 100},
                state,
                _UpdateBlobCtx(
                    catalog=catalog,
                    session_engine=self.engine,
                    session_id=self.session_id,
                    **_verbatim_blob_context(self.engine, self.session_id, "x" * 100),
                ),
            )

        # Headline is the primary RuntimeError.
        assert type(exc_info.value) is RuntimeError, f"Unexpected exception type: got {type(exc_info.value).__name__}"
        # No rollback write was performed — the tempfile carries the new
        # bytes but storage_path was never written.
        assert write_bytes_calls_to_storage[0] == 0, (
            f"Pre-replace failure should not trigger a storage_path rollback write; "
            f"got {write_bytes_calls_to_storage[0]} writes to {target_path_str}"
        )
        # No add_note diagnostic — no divergence to record.
        notes = getattr(exc_info.value, "__notes__", [])
        assert not any("Rollback failed" in n for n in notes), f"Spurious rollback note on pre-replace failure: {notes!r}"
        # File contents intact.
        assert self.storage_path.read_bytes() == self.original_content
        # Tempfile cleaned up.
        leftovers = [p for p in self.storage_path.parent.iterdir() if p != self.storage_path]
        assert leftovers == [], f"Tempfile leaked: {leftovers}"

    def test_clean_db_failure_before_replace_leaves_no_residue(self) -> None:
        """Pre-replace DB failure: file intact, no note, no tempfile residue.

        Companion to the test above — same invariant but with the
        cleanest possible setup (no write_bytes tripwire) so a future
        reader can see the happy-path exit shape in isolation.
        """
        from unittest.mock import patch

        from elspeth.web.composer.tools import _execute_update_blob

        primary_message = "primary-db-fault-clean-exit"

        def _raise_primary(*_args: Any, **_kwargs: Any) -> str | None:
            raise RuntimeError(primary_message)

        state = _empty_state()
        catalog = _mock_catalog()

        with (
            patch("elspeth.web.composer.tools.blobs._check_blob_quota", side_effect=_raise_primary),
            pytest.raises(RuntimeError, match=primary_message) as exc_info,
        ):
            from elspeth.web.composer.tools._common import ToolContext as _UpdateBlobCtx

            _execute_update_blob(
                {"blob_id": self.blob_id, "content": "x" * 100},
                state,
                _UpdateBlobCtx(
                    catalog=catalog,
                    session_engine=self.engine,
                    session_id=self.session_id,
                    **_verbatim_blob_context(self.engine, self.session_id, "x" * 100),
                ),
            )

        assert self.storage_path.read_bytes() == self.original_content
        notes = getattr(exc_info.value, "__notes__", [])
        assert not any("Rollback failed" in n for n in notes), f"Spurious rollback note attached on clean DB failure: {notes!r}"
        leftovers = [p for p in self.storage_path.parent.iterdir() if p != self.storage_path]
        assert leftovers == [], f"Tempfile leaked: {leftovers}"


class TestSessionBlobLockRegistry:
    """``_session_blob_lock`` must return a stable lock per session_id.

    The lock identity is the contract the ``_execute_update_blob``
    critical section depends on: two threads asking for the same
    session_id's lock must receive the SAME ``threading.Lock`` instance
    so acquiring it in one thread blocks the other.  A broken registry
    that returned fresh locks on every call would offer no
    serialisation at all — correctness would silently regress to the
    pre-I4 race.
    """

    def test_same_session_returns_identical_lock(self) -> None:
        """Two lookups for the same session_id must return the same lock."""
        from elspeth.web.composer.tools import _session_blob_lock

        session_id = "test-session-identity"
        first = _session_blob_lock(session_id)
        second = _session_blob_lock(session_id)
        assert first is second, (
            "Session lock registry returned a DIFFERENT lock for the same session_id; two concurrent updaters would not serialise."
        )

    def test_different_sessions_return_distinct_locks(self) -> None:
        """Different session_ids must map to different locks (no global bottleneck)."""
        from elspeth.web.composer.tools import _session_blob_lock

        lock_a = _session_blob_lock("session-A")
        lock_b = _session_blob_lock("session-B")
        assert lock_a is not lock_b, (
            "Session lock registry returned the SAME lock for different session_ids; unrelated sessions would contend."
        )

    def test_concurrent_lookups_converge_on_single_lock(self) -> None:
        """Under concurrent first-access, all threads must receive the same lock.

        Regression guard for the double-checked-locking implementation
        in ``_session_blob_lock``.  Without the registry mutex, two
        threads asking for a not-yet-present session_id at the same
        time could each install a different lock and half the callers
        would serialise against one instance while the other half
        serialise against the other — the race the I4 fix closes would
        persist across threads partitioned by lock identity.
        """
        import threading as stdlib_threading
        from uuid import uuid4

        from elspeth.web.composer.tools import _session_blob_lock

        session_id = f"concurrent-{uuid4()}"
        start = stdlib_threading.Event()
        locks: list[Any] = []
        lock_guard = stdlib_threading.Lock()

        def worker() -> None:
            start.wait()
            lock = _session_blob_lock(session_id)
            with lock_guard:
                locks.append(lock)

        threads = [stdlib_threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join()

        assert len(locks) == 16
        assert all(lock is locks[0] for lock in locks), (
            "Concurrent _session_blob_lock callers received distinct lock instances; "
            "the registry mutex is missing or double-checked locking is broken."
        )


class TestUpdateBlobSessionLockSerialisation:
    """_execute_update_blob must acquire the session lock BEFORE _sync_get_blob.

    This is the I4 fix: the read→write→commit critical section must be
    atomic across concurrent composer-tool callers on the same session.
    Holding the session lock externally from the test must block the
    tool call entirely — if the tool bypasses the lock, the worker
    thread completes while the main thread still holds the mutex,
    revealing the race.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = f"lock-serialise-{uuid4()}"
        self.blob_id = str(uuid4())
        self.data_dir = str(tmp_path)
        now = datetime.now(UTC)

        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_data.csv"
        self.original_content = b"orig"
        self.storage_path.write_bytes(self.original_content)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.original_content),
                    content_hash=_STUB_SHA256,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def test_update_blob_blocks_when_session_lock_is_held(self) -> None:
        """Worker must NOT complete while the main thread holds the lock.

        Strategy: acquire the session lock externally, spawn a worker
        that calls update_blob, assert the worker is still alive after
        a short wait.  Releasing the lock unblocks the worker, which
        must then complete and produce a successful update.  A regression
        that skipped the lock would make the worker complete before the
        lock is released — the short-wait assertion catches that.

        Inverted-flake guard: a ``started`` event set at the very top
        of the worker body distinguishes "blocked by the lock as
        intended" from "worker thread never ran" (scheduler stall,
        import failure, etc.).  Without the probe, both cases look
        identical (completed.wait returns False) and a broken test
        would falsely pass.
        """
        import threading as stdlib_threading

        from elspeth.web.composer.tools import _session_blob_lock

        lock = _session_blob_lock(self.session_id)
        started = stdlib_threading.Event()
        completed = stdlib_threading.Event()
        result_holder: list[Any] = []

        def worker() -> None:
            started.set()
            try:
                result = execute_tool(
                    "update_blob",
                    {"blob_id": self.blob_id, "content": "new-content-from-worker"},
                    _empty_state(),
                    _mock_catalog(),
                    session_engine=self.engine,
                    session_id=self.session_id,
                    **_verbatim_blob_context(self.engine, self.session_id, "new-content-from-worker"),
                )
                result_holder.append(result)
            finally:
                completed.set()

        lock.acquire()
        try:
            t = stdlib_threading.Thread(target=worker, daemon=True)
            t.start()
            # Probe first: the worker must actually enter its body
            # before we interpret ``completed.wait`` returning False as
            # "lock-blocked."  A thread that never scheduled would
            # satisfy the completed-wait check for the wrong reason.
            assert started.wait(timeout=2.0), "worker thread never entered its body"
            # While we hold the lock, the worker MUST NOT complete — if
            # it does, the tool bypassed the session lock. Keep a full
            # second of slack here: under xdist load the worker thread may
            # take longer than a few hundred milliseconds to reach the
            # lock acquisition point even though the locking contract is
            # correct.
            blocked = not completed.wait(timeout=1.0)
            assert blocked, (
                "update_blob completed while the session lock was held externally; "
                "the tool did not acquire _session_blob_lock before _sync_get_blob, "
                "reopening the I4 file/DB rollback race."
            )
        finally:
            lock.release()

        assert completed.wait(timeout=2.0), "update_blob did not complete within 2s after session lock was released"
        t.join(timeout=2.0)
        assert not t.is_alive(), "worker thread failed to exit after update completed"
        assert result_holder, "worker did not produce a result"
        assert result_holder[0].success is True, f"Update failed after lock release: {result_holder[0].data}"
        assert self.storage_path.read_bytes() == b"new-content-from-worker"


class TestUpdateBlobQuotaRollbackDivergence:
    """Quota breach must return ``_failure_result`` without any file mutation.

    Pre-atomic-rename, ``_execute_update_blob`` overwrote
    ``storage_path`` BEFORE the DB transaction; a quota breach inside
    the transaction therefore required a rollback write, and a
    rollback-write OSError was surfaced via a RuntimeError with
    add_note divergence discipline (the I5 fix).

    Post atomic-rename (bug_004), the file is written to a sibling
    tempfile and swapped in via ``os.replace`` only AFTER the quota
    check has passed.  A quota breach therefore happens before any
    file mutation — no rollback, no RuntimeError, no add_note; the
    caller simply sees a ``ToolResult(success=False, ...)`` carrying
    the quota message.  The divergence-on-rollback-OSError discipline
    remains in the code as a defensive guardrail for the narrow
    post-replace commit-failure window, but it is no longer reachable
    via the quota path.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        self.data_dir = str(tmp_path)
        now = datetime.now(UTC)

        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_data.csv"
        self.original_content = b"pre-quota-bytes"
        self.storage_path.write_bytes(self.original_content)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.original_content),
                    content_hash=_STUB_SHA256,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def test_quota_breach_returns_failure_without_touching_storage(self) -> None:
        """Quota failure returns ToolResult(success=False) with storage intact.

        Under the atomic-rename design the quota check runs BEFORE
        ``os.replace``, so a quota breach leaves storage_path exactly
        as it was.  No rollback write is needed, no RuntimeError is
        raised, and no add_note is attached — the LLM simply sees a
        failure result describing the quota exhaustion.

        Tripwire: patches ``Path.write_bytes`` to fail on any write to
        storage_path so an accidental regression to "write-first then
        rollback" would surface as an ENOSPC-like error instead of a
        clean quota failure.
        """
        from unittest.mock import patch

        real_write_bytes = Path.write_bytes
        target_path_str = str(self.storage_path)
        tripwire_hits: list[str] = []

        def _tripwire_write_bytes(path_self: Path, data: bytes) -> int:
            if str(path_self) == target_path_str:
                tripwire_hits.append("storage_path was written pre-replace")
                raise OSError(28, "Tripwire: pre-replace write to storage_path not allowed")
            return real_write_bytes(path_self, data)

        state = _empty_state()
        catalog = _mock_catalog()

        # Quota 10 bytes; new content 100 bytes → delta 85 exceeds quota.
        with (
            patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", 10),
            patch.object(Path, "write_bytes", _tripwire_write_bytes),
        ):
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "x" * 100},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "x" * 100),
            )

        # Clean failure result — no exception, no divergence.
        assert result.success is False, f"Expected quota failure result, got {result!r}"
        assert "quota" in result.data["error"].lower(), f"Quota failure message missing from error: {result.data['error']!r}"
        # Tripwire must not have fired — no write to storage_path.
        assert tripwire_hits == [], f"Pre-replace write to storage_path detected (atomic-rename regression): {tripwire_hits}"
        # Storage unchanged.
        assert self.storage_path.read_bytes() == self.original_content
        # Tempfile cleaned up in finally.
        leftovers = [p for p in self.storage_path.parent.iterdir() if p != self.storage_path]
        assert leftovers == [], f"Tempfile leaked after quota breach: {leftovers}"

    def test_quota_rollback_success_returns_failure_result_not_exception(self) -> None:
        """When rollback succeeds on quota path, callers still get a ToolResult.

        Regression guard: the quota-exceeded contract is that callers
        receive a ``ToolResult(success=False, ...)`` — NOT a raised
        exception — when the quota is breached AND the rollback
        succeeds.  The I5 fix adds the divergence-on-rollback-failure
        path without changing the happy-path shape.
        """
        from unittest.mock import patch

        state = _empty_state()
        catalog = _mock_catalog()

        with patch("elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES", 10):
            result = execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "x" * 100},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **_verbatim_blob_context(self.engine, self.session_id, "x" * 100),
            )

        assert result.success is False, "Quota-exceeded must return failure, not success"
        assert "quota" in result.data["error"].lower()
        # File must be restored to original content.
        assert self.storage_path.read_bytes() == self.original_content, (
            "File was not rolled back after quota-exceeded with successful rollback"
        )


# ---------------------------------------------------------------------------
# Secret tool tests — composer-level secret reference wiring
# ---------------------------------------------------------------------------


class TestSecretTools:
    """Secret reference composition tools: discovery, validation, and wiring.

    Security contracts tested:
    - Secret tools fail without secret_service (no ambient access)
    - list_secret_refs never returns plaintext values
    - validate_secret_ref returns availability status
    - wire_secret_ref sets a secret_ref marker in source options
    """

    def _mock_secret_service(self) -> MagicMock:
        from elspeth.contracts.secrets import SecretInventoryItem

        svc = MagicMock()
        svc.list_refs.return_value = [
            SecretInventoryItem(name="OPENROUTER_API_KEY", scope="user", available=True),
            SecretInventoryItem(name="DB_PASSWORD", scope="server", available=True),
        ]
        svc.has_ref.return_value = True
        return svc

    def test_list_secret_refs_without_service_returns_failure(self) -> None:
        """Secret tools with no secret_service must fail — no ambient access."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("list_secret_refs", {}, state, catalog)
        assert result.success is False

    def test_list_secret_refs_returns_inventory(self) -> None:
        """list_secret_refs returns inventory items without values."""
        state = _empty_state()
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        result = execute_tool(
            "list_secret_refs",
            {},
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )
        assert result.success is True
        assert len(result.data) == 2
        # Ensure no value field leaked
        for item in result.data:
            assert "value" not in item

    def test_list_secret_refs_returns_unavailability_reason(self) -> None:
        """Unavailable secret refs expose structural reasons, never values."""
        from elspeth.contracts.secrets import SecretInventoryItem

        state = _empty_state()
        catalog = _mock_catalog()
        svc = MagicMock()
        svc.list_refs.return_value = [
            SecretInventoryItem(
                name="OPENROUTER_API_KEY",
                scope="server",
                available=False,
                source_kind="env",
                reason="fingerprint_resolver_not_configured",
            )
        ]

        result = execute_tool(
            "list_secret_refs",
            {},
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )

        assert result.success is True
        assert len(result.data) == 1
        assert dict(result.data[0]) == {
            "name": "OPENROUTER_API_KEY",
            "scope": "server",
            "available": False,
            "source_kind": "env",
            "reason": "fingerprint_resolver_not_configured",
        }
        assert "value" not in repr(result.to_dict())

    def test_validate_secret_ref_returns_availability(self) -> None:
        """validate_secret_ref returns name and availability status."""
        state = _empty_state()
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        result = execute_tool(
            "validate_secret_ref",
            {"name": "OPENROUTER_API_KEY"},
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )
        assert result.success is True
        assert result.data["name"] == "OPENROUTER_API_KEY"
        assert result.data["available"] is True

    def test_validate_secret_ref_returns_unavailability_reason(self) -> None:
        """Known but unavailable secret refs carry the inventory reason."""
        from elspeth.contracts.secrets import SecretInventoryItem

        state = _empty_state()
        catalog = _mock_catalog()
        svc = MagicMock()
        svc.has_ref.return_value = False
        svc.list_refs.return_value = [
            SecretInventoryItem(
                name="OPENROUTER_API_KEY",
                scope="server",
                available=False,
                source_kind="env",
                reason="fingerprint_resolver_not_configured",
            )
        ]

        result = execute_tool(
            "validate_secret_ref",
            {"name": "OPENROUTER_API_KEY"},
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )

        assert result.success is True
        assert dict(result.data) == {
            "name": "OPENROUTER_API_KEY",
            "available": False,
            "scope": "server",
            "source_kind": "env",
            "reason": "fingerprint_resolver_not_configured",
        }

    def test_validate_secret_ref_uses_bounded_resolvability_check_for_inventory_hits(self) -> None:
        """validate_secret_ref must not trust cheap inventory availability."""
        from elspeth.contracts.secrets import SecretInventoryItem

        state = _empty_state()
        catalog = _mock_catalog()
        svc = MagicMock()
        svc.has_ref.return_value = False
        svc.list_refs.return_value = [
            SecretInventoryItem(
                name="OPENROUTER_API_KEY",
                scope="user",
                available=True,
                source_kind="user_store",
            )
        ]

        result = execute_tool(
            "validate_secret_ref",
            {"name": "OPENROUTER_API_KEY"},
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )

        svc.has_ref.assert_called_once_with("test-user", "OPENROUTER_API_KEY")
        assert result.success is True
        assert dict(result.data) == {
            "name": "OPENROUTER_API_KEY",
            "available": False,
            "scope": "user",
            "source_kind": "user_store",
            "reason": "value_decryption_failed",
        }

    def test_validate_secret_ref_without_service_returns_failure(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "validate_secret_ref",
            {"name": "OPENROUTER_API_KEY"},
            state,
            catalog,
        )
        assert result.success is False

    def test_wire_secret_ref_sets_marker_in_source_options(self) -> None:
        """wire_secret_ref patches source options with a secret_ref marker."""
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        # First set a source
        state = _empty_state()
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert r1.success is True
        # Now wire a secret into the source
        r2 = execute_tool(
            "wire_secret_ref",
            {
                "name": "OPENROUTER_API_KEY",
                "target": "source",
                "option_key": "api_key",
            },
            r1.updated_state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )
        assert r2.success is True
        assert _default_source(r2.updated_state) is not None

        opts = deep_thaw(_default_source(r2.updated_state).options)
        assert opts["api_key"] == {"secret_ref": "OPENROUTER_API_KEY"}
        # Original options preserved
        assert opts["path"] == "/data/in.csv"

    def test_wire_secret_ref_rejects_source_non_credential_field(self) -> None:
        """wire_secret_ref must enforce the same placement policy as set_source."""
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        state = _empty_state()
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert r1.success is True

        result = execute_tool(
            "wire_secret_ref",
            {
                "name": "OPENROUTER_API_KEY",
                "target": "source",
                "option_key": "path",
            },
            r1.updated_state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )

        assert result.success is False
        assert result.updated_state is r1.updated_state
        assert "csv" in result.data["error"]
        assert "path" in result.data["error"]
        assert "OPENROUTER_API_KEY" in result.data["error"]
        assert "only credential-bearing fields" in result.data["error"]

    def test_wire_secret_ref_rejects_node_non_credential_field(self) -> None:
        """wire_secret_ref must enforce placement policy for node options."""
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        state = _empty_state()
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "classify",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": _llm_options_with_api_key({"secret_ref": "OPENROUTER_API_KEY"}),
            },
            state,
            catalog,
        )
        assert r1.success is True

        result = execute_tool(
            "wire_secret_ref",
            {
                "name": "OPENROUTER_API_KEY",
                "target": "node",
                "target_id": "classify",
                "option_key": "template",
            },
            r1.updated_state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )

        assert result.success is False
        assert result.updated_state is r1.updated_state
        assert "llm" in result.data["error"]
        assert "template" in result.data["error"]
        assert "OPENROUTER_API_KEY" in result.data["error"]
        assert "only credential-bearing fields" in result.data["error"]

    def test_wire_secret_ref_rejects_output_non_credential_field(self) -> None:
        """wire_secret_ref must enforce placement policy for output options."""
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        state = _empty_state()
        r1 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert r1.success is True

        result = execute_tool(
            "wire_secret_ref",
            {
                "name": "OPENROUTER_API_KEY",
                "target": "output",
                "target_id": "main",
                "option_key": "path",
            },
            r1.updated_state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )

        assert result.success is False
        assert result.updated_state is r1.updated_state
        assert "csv" in result.data["error"]
        assert "path" in result.data["error"]
        assert "OPENROUTER_API_KEY" in result.data["error"]
        assert "only credential-bearing fields" in result.data["error"]

    def test_wire_secret_ref_without_service_returns_failure(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "wire_secret_ref",
            {
                "name": "OPENROUTER_API_KEY",
                "target": "source",
                "option_key": "api_key",
            },
            state,
            catalog,
        )
        assert result.success is False

    def test_wire_secret_ref_nonexistent_ref_fails(self) -> None:
        """wire_secret_ref fails if the secret ref doesn't exist."""
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        svc.has_ref.return_value = False
        state = _empty_state()
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "wire_secret_ref",
            {
                "name": "NONEXISTENT",
                "target": "source",
                "option_key": "api_key",
            },
            r1.updated_state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )
        assert r2.success is False

    def test_wire_secret_ref_gets_prior_validation(self) -> None:
        """Secret mutation tools must populate prior_validation for validation delta."""
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        state = _empty_state()
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert r1.success is True
        r2 = execute_tool(
            "wire_secret_ref",
            {
                "name": "OPENROUTER_API_KEY",
                "target": "source",
                "option_key": "api_key",
            },
            r1.updated_state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )
        assert r2.success is True
        assert r2.prior_validation is not None
        d = r2.to_dict()
        assert "validation_delta" in d

    def _build_state_with_source(self, catalog: Any) -> CompositionState:
        """Helper: build state with a source for secret tool tests."""
        r = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            _empty_state(),
            catalog,
        )
        assert r.success is True
        return r.updated_state

    def test_wire_secret_ref_threaded_prior_used(self) -> None:
        """Threaded prior_validation is reused by secret mutation tools (identity check).

        execute_tool dispatches secret mutations through a separate branch from
        standard mutations. Both branches must honour the prior_validation kwarg.
        """
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        state = self._build_state_with_source(catalog)
        threaded = state.validate()
        result = execute_tool(
            "wire_secret_ref",
            {
                "name": "OPENROUTER_API_KEY",
                "target": "source",
                "option_key": "api_key",
            },
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
            prior_validation=threaded,
        )
        assert result.success is True
        assert result.prior_validation is threaded

    def test_wire_secret_ref_threaded_prior_produces_correct_delta(self) -> None:
        """Threaded and fresh prior_validation produce identical deltas for secret tools."""
        catalog = _mock_catalog()
        svc = self._mock_secret_service()
        state = self._build_state_with_source(catalog)
        secret_args = {
            "name": "OPENROUTER_API_KEY",
            "target": "source",
            "option_key": "api_key",
        }
        # Fresh (no threading)
        result_fresh = execute_tool(
            "wire_secret_ref",
            secret_args,
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
        )
        # Threaded
        threaded = state.validate()
        result_threaded = execute_tool(
            "wire_secret_ref",
            secret_args,
            state,
            catalog,
            secret_service=svc,
            user_id="test-user",
            prior_validation=threaded,
        )
        delta_fresh = result_fresh.to_dict()["validation_delta"]
        delta_threaded = result_threaded.to_dict()["validation_delta"]
        assert delta_fresh == delta_threaded


class TestSecretToolsArgumentValidation:
    """Tier-3 shape contract for the secret-ref handlers.

    Pairs with ``TestSecretTools`` (semantic / state-mutation paths).  These
    tests exercise the Pydantic argument-model layer that converts shape
    failures into :class:`ToolArgumentError` — the same dispatch-layer
    contract sources/blobs/outputs/sessions already satisfy.  Without this
    layer, a malformed LLM call (wrong type, extra field, unknown ``target``
    enum) would escape ``execute_tool`` as a bare exception and be
    laundered by the compose loop's catch-all into
    :class:`ComposerPluginCrashError` → HTTP 500, denying the LLM the
    recoverable ARG_ERROR payload it needs to self-correct.
    """

    def _svc(self) -> MagicMock:
        from elspeth.contracts.secrets import SecretInventoryItem

        svc = MagicMock()
        svc.list_refs.return_value = [
            SecretInventoryItem(name="OPENROUTER_API_KEY", scope="user", available=True),
        ]
        svc.has_ref.return_value = True
        return svc

    def test_validate_secret_ref_wrong_type_for_name(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "validate_secret_ref",
                {"name": 42},
                state,
                catalog,
                secret_service=self._svc(),
                user_id="test-user",
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_validate_secret_ref_rejects_extra_field(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "validate_secret_ref",
                {"name": "OPENROUTER_API_KEY", "bogus": "value"},
                state,
                catalog,
                secret_service=self._svc(),
                user_id="test-user",
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_wire_secret_ref_wrong_type_for_name(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "wire_secret_ref",
                {"name": 42, "target": "source", "option_key": "api_key"},
                state,
                catalog,
                secret_service=self._svc(),
                user_id="test-user",
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_wire_secret_ref_unknown_target_enum(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "wire_secret_ref",
                {"name": "OPENROUTER_API_KEY", "target": "global", "option_key": "api_key"},
                state,
                catalog,
                secret_service=self._svc(),
                user_id="test-user",
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_wire_secret_ref_rejects_extra_field(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool(
                "wire_secret_ref",
                {
                    "name": "OPENROUTER_API_KEY",
                    "target": "source",
                    "option_key": "api_key",
                    "bogus": "value",
                },
                state,
                catalog,
                secret_service=self._svc(),
                user_id="test-user",
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)


# ---------------------------------------------------------------------------
# Merge-patch helper tests
# ---------------------------------------------------------------------------


class TestMergePatch:
    def test_merge_patch_overwrites(self) -> None:
        result = _apply_merge_patch({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_merge_patch_adds(self) -> None:
        result = _apply_merge_patch({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_patch_deletes_null(self) -> None:
        result = _apply_merge_patch({"a": 1, "b": 2}, {"b": None})
        assert result == {"a": 1}
        assert "b" not in result

    def test_merge_patch_null_for_absent_key_is_noop(self) -> None:
        # Delete-if-present semantics: setting an absent key to None must be a
        # silent no-op, never a KeyError. This pins the edge case that
        # distinguishes the delete-if-present idiom from an unguarded ``del``.
        result = _apply_merge_patch({"a": 1}, {"b": None})
        assert result == {"a": 1}
        assert "b" not in result

    def test_merge_patch_preserves_unmentioned(self) -> None:
        result = _apply_merge_patch({"a": 1, "b": 2}, {"a": 3})
        assert result == {"a": 3, "b": 2}

    def test_merge_patch_empty_patch(self) -> None:
        result = _apply_merge_patch({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_patch_does_not_mutate_target(self) -> None:
        proxy = MappingProxyType({"a": 1})
        result = _apply_merge_patch(proxy, {"a": 2})
        # Original proxy is unchanged
        assert proxy["a"] == 1
        assert result == {"a": 2}


# ---------------------------------------------------------------------------
# patch_source_options tool tests
# ---------------------------------------------------------------------------


class TestPatchSourceOptions:
    def _state_with_source(self, options: dict[str, Any]) -> CompositionState:
        state = _empty_state()
        catalog = _mock_catalog()
        merged = {"schema": {"mode": "observed"}, **options}
        r = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": merged,
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert r.success is True
        return r.updated_state

    def test_patch_source_options_updates_key(self) -> None:
        state = self._state_with_source({"path": "/a"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"path": "/b"}},
            state,
            catalog,
        )
        assert result.success is True
        assert _default_source(result.updated_state) is not None

        opts = deep_thaw(_default_source(result.updated_state).options)
        assert opts["path"] == "/b"

    def test_patch_source_options_adds_key(self) -> None:
        state = self._state_with_source({"path": "/a"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"encoding": "utf-8"}},
            state,
            catalog,
        )
        assert result.success is True

        assert _default_source(result.updated_state) is not None
        opts = deep_thaw(_default_source(result.updated_state).options)
        assert opts["path"] == "/a"
        assert opts["encoding"] == "utf-8"

    def test_patch_source_options_rejects_literal_credential_value_without_mutating_state(self) -> None:
        state = self._state_with_source({"path": "/a"})
        catalog = _mock_catalog()
        literal = "literal-source-key-for-test"

        result = execute_tool(
            "patch_source_options",
            {"patch": {"api_key": literal}},
            state,
            catalog,
        )

        _assert_secret_wiring_contract_failure(
            result,
            state,
            literal_value=literal,
            field="source:api_key",
        )

    def test_patch_source_options_deletes_key(self) -> None:
        state = self._state_with_source({"path": "/a", "encoding": "utf-8"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"encoding": None}},
            state,
            catalog,
        )
        assert result.success is True

        assert _default_source(result.updated_state) is not None
        opts = deep_thaw(_default_source(result.updated_state).options)
        assert opts["path"] == "/a"
        assert "encoding" not in opts

    def test_patch_source_options_no_source_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"path": "/b"}},
            state,
            catalog,
        )
        assert result.success is False
        assert "No source" in result.data["error"]
        assert result.updated_state.version == 1

    def _blob_backed_state(self) -> CompositionState:
        """Build a state with a blob-backed source directly.

        Bypasses ``set_source`` (which now rejects manual ``blob_ref``
        injection per elspeth-07089fbaa3) by constructing the
        ``CompositionState`` and ``SourceSpec`` from the dataclass
        primitives.  This is the post-fix shape produced by
        ``set_source_from_blob`` at runtime.
        """
        return CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="t1",
                options={
                    "blob_ref": "abc123",
                    "path": "/canon/abc123_x.csv",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def test_patch_source_options_rejects_path_patch_on_blob_backed_source(self) -> None:
        """Closes elspeth-07089fbaa3 (write defense, branch b).

        Once a source is bound to a blob via ``set_source_from_blob``, the
        ``path`` is the blob's canonical ``storage_path`` and is not
        patchable.  Allowing a ``path`` patch lets the LLM persist a path
        that disagrees with the bound blob, breaking runtime path
        resolution and composer/runtime agreement.  Re-binding to a
        different blob requires ``set_source_from_blob`` (or
        ``clear_source`` first), not a path patch.

        Bug-verification protocol (cf.
        ``tests/integration/pipeline/test_composer_runtime_agreement.py``
        module docstring lines 76-88): manually revert the
        ``if "blob_ref" in state.source.options:`` block in
        ``_execute_patch_source_options`` (src/elspeth/web/composer/tools.py)
        and confirm this test fails with ``result.success is True``.
        Then restore.
        """
        state = self._blob_backed_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"path": "/canon/abc123_x.csv"}},
            state,
            catalog,
        )
        assert result.success is False
        assert "blob-backed source" in result.data["error"]

    def test_patch_source_options_rejects_blob_ref_patch_on_plain_source(self) -> None:
        """Manual blob identity injection is rejected even before a source is blob-backed."""
        state = self._state_with_source({"path": "/a.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"blob_ref": "def456"}},
            state,
            catalog,
        )
        assert result.success is False
        assert "Cannot patch 'blob_ref'" in result.data["error"]
        assert _default_source(result.updated_state) is not None
        opts = deep_thaw(_default_source(result.updated_state).options)
        assert "blob_ref" not in opts
        assert opts["path"] == "/a.csv"

    def test_patch_source_options_rejects_blob_ref_patch_on_blob_backed_source(self) -> None:
        """Closes elspeth-07089fbaa3 (write defense, blob_ref re-bind via patch).

        ``blob_ref`` is part of the immutable binding; replacing it via a
        patch would silently re-target the source to a different blob
        without re-deriving the canonical ``storage_path``.  Re-binding
        requires ``set_source_from_blob``.
        """
        state = self._blob_backed_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"blob_ref": "def456"}},
            state,
            catalog,
        )
        assert result.success is False
        assert "Cannot patch 'blob_ref'" in result.data["error"]

    def test_patch_source_options_allows_unrelated_keys_on_blob_backed_source(self) -> None:
        """Closes elspeth-07089fbaa3 (write defense — narrow rejection).

        The (path, blob_ref) lock must not over-reach: patches that touch
        only schema/encoding/etc. on a blob-backed source remain valid,
        because they don't break the blob binding.  Without this the
        defense-in-depth would block legitimate plugin-option tuning.
        """
        state = self._blob_backed_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_source_options",
            {"patch": {"encoding": "utf-8"}},
            state,
            catalog,
        )
        assert result.success is True
        assert _default_source(result.updated_state) is not None
        opts = deep_thaw(_default_source(result.updated_state).options)
        assert opts["encoding"] == "utf-8"
        assert opts["path"] == "/canon/abc123_x.csv"
        assert opts["blob_ref"] == "abc123"


# ---------------------------------------------------------------------------
# patch_node_options tool tests
# ---------------------------------------------------------------------------


class TestPatchNodeOptions:
    def _state_with_node(self, options: dict[str, Any]) -> CompositionState:
        state = _empty_state()
        catalog = _mock_catalog()
        r = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "source_out",
                "on_success": "main",
                "options": options,
            },
            state,
            catalog,
        )
        assert r.success is True
        return r.updated_state

    def test_patch_node_options_updates_key(self) -> None:
        state = self._state_with_node({"schema": {"mode": "observed"}, "required_input_fields": ["old"]})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_node_options",
            {"node_id": "t1", "patch": {"required_input_fields": ["new"]}},
            state,
            catalog,
        )
        assert result.success is True
        node = result.updated_state.nodes[0]
        assert node.id == "t1"

        opts = deep_thaw(node.options)
        assert opts["required_input_fields"] == ["new"]
        # Other node fields preserved
        assert node.node_type == "transform"
        assert node.plugin == "passthrough"

    def test_patch_node_options_rejects_literal_credential_value_without_mutating_state(self) -> None:
        state = self._state_with_node({"schema": {"mode": "observed"}})
        catalog = _mock_catalog()
        literal = "literal-node-key-for-test"

        result = execute_tool(
            "patch_node_options",
            {"node_id": "t1", "patch": {"api_key": literal}},
            state,
            catalog,
        )

        _assert_secret_wiring_contract_failure(
            result,
            state,
            literal_value=literal,
            field="t1:api_key",
        )

    def test_patch_node_options_rejects_on_error_with_routing_tool_guidance(self) -> None:
        """on_error is a node routing field, not a plugin option patch."""
        state = _empty_state()
        catalog = _mock_catalog()
        with_node = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            },
            state,
            catalog,
        )
        assert with_node.success is True

        result = execute_tool(
            "patch_node_options",
            {"node_id": "t1", "patch": {"on_error": "llm_failures"}},
            with_node.updated_state,
            catalog,
        )

        assert result.success is False
        assert result.updated_state is with_node.updated_state
        assert result.updated_state.nodes[0].on_error == "discard"
        assert "on_error is a node-level routing field" in result.data["error"]
        assert "upsert_edge" in result.data["error"]
        assert "edge_type='on_error'" in result.data["error"]
        assert "Extra inputs are not permitted" not in result.data["error"]

    def test_patch_node_options_unknown_node_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_node_options",
            {"node_id": "nonexistent", "patch": {"field": "value"}},
            state,
            catalog,
        )
        assert result.success is False
        assert "nonexistent" in result.data["error"]


# ---------------------------------------------------------------------------
# patch_output_options tool tests
# ---------------------------------------------------------------------------


class TestPatchOutputOptions:
    def _state_with_output(self, options: dict[str, Any]) -> CompositionState:
        state = _empty_state()
        catalog = _mock_catalog()
        merged = {"schema": {"mode": "observed"}, **options}
        r = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": merged,
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert r.success is True
        return r.updated_state

    def test_patch_output_options_updates_key(self) -> None:
        state = self._state_with_output({"path": "/old.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"path": "/new.csv"}},
            state,
            catalog,
        )
        assert result.success is True
        output = result.updated_state.outputs[0]

        opts = deep_thaw(output.options)
        assert opts["path"] == "/new.csv"

    def test_patch_output_options_rejects_literal_credential_value_without_mutating_state(self) -> None:
        state = self._state_with_output({"path": "/old.csv"})
        catalog = _mock_catalog()
        literal = "literal-output-password-for-test"

        result = execute_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"password": literal}},
            state,
            catalog,
        )

        _assert_secret_wiring_contract_failure(
            result,
            state,
            literal_value=literal,
            field="main:password",
        )

    def test_patch_output_options_unknown_sink_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {"sink_name": "nonexistent", "patch": {"path": "/x.csv"}},
            state,
            catalog,
        )
        assert result.success is False
        assert "nonexistent" in result.data["error"]


# ---------------------------------------------------------------------------
# Patch output path security (Finding 1: 3554012f39)
# ---------------------------------------------------------------------------


class TestPatchOutputPathSecurity:
    """S2: Sink path allowlist — patched output paths must be under allowed directories.

    Mirrors TestSetSourcePathSecurity but for the sink/output side.
    _validate_sink_path() must be called after merge-patching output options.
    """

    def _state_with_output(self, options: dict[str, Any]) -> CompositionState:
        state = _empty_state()
        catalog = _mock_catalog()
        merged = {"schema": {"mode": "observed"}, **options}
        r = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": merged,
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert r.success is True
        return r.updated_state

    def test_path_outside_allowlist_rejected(self) -> None:
        state = self._state_with_output({"path": "/data/outputs/ok.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"path": "/etc/passwd"}},
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False
        assert "path" in result.data["error"].lower()

    def test_traversal_attack_rejected(self) -> None:
        state = self._state_with_output({"path": "/data/outputs/ok.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"path": "/data/outputs/../../etc/passwd"}},
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_file_key_also_validated(self) -> None:
        state = self._state_with_output({"path": "/data/outputs/ok.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"file": "/tmp/evil.csv"}},
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_persist_directory_key_also_validated(self) -> None:
        from elspeth.web.composer.tools._common import _validate_sink_path

        error = _validate_sink_path({"persist_directory": "/tmp/elspeth-outside"}, data_dir="/data")
        assert error is not None
        assert "persist_directory" in error

    def test_file_key_traversal_rejected(self) -> None:
        state = self._state_with_output({"path": "/data/outputs/ok.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"file": "/data/outputs/../../etc/shadow"}},
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_relative_path_under_outputs_accepted(self) -> None:
        state = self._state_with_output({"path": "/data/outputs/ok.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {
                "sink_name": "main",
                "patch": {
                    "path": "outputs/result.csv",
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is True

    def test_absolute_path_under_allowed_dir_accepted(self) -> None:
        state = self._state_with_output({"path": "/data/outputs/ok.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {
                "sink_name": "main",
                "patch": {
                    "path": "/data/outputs/subdir/out.csv",
                    "mode": "write",
                    "collision_policy": "fail_if_exists",
                },
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is True

    def test_data_dir_none_skips_validation(self) -> None:
        """When data_dir is not configured, any path is accepted."""
        state = self._state_with_output({"path": "/anywhere/file.csv"})
        catalog = _mock_catalog()
        result = execute_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"path": "/etc/passwd"}},
            state,
            catalog,
            data_dir=None,
        )
        assert result.success is True


class TestTransformProviderConfigPathSecurity:
    """S2: nested transform provider_config persist_directory must be confined.

    RAG retrieval transforms carry a local Chroma persist_directory under
    options.provider_config. Mirrors the sink-path guard: confined to the
    allowed output directories.
    """

    @staticmethod
    def _catalog_with_rag() -> MagicMock:
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="rag_retrieval",
                description="RAG retrieval transform",
                plugin_type="transform",
                config_fields=[],
            ),
        ]
        return catalog

    @staticmethod
    def _rag_options(persist_directory: str) -> dict[str, Any]:
        return {
            "provider": "chroma",
            "provider_config": {"persist_directory": persist_directory, "collection": "docs"},
            "schema": {"mode": "observed"},
            "output_prefix": "rag_",
            "query_field": "text",
        }

    @staticmethod
    def _azure_search_managed_identity_options() -> dict[str, Any]:
        return {
            "provider": "azure_search",
            "provider_config": {
                "endpoint": "https://tenant-b.search.windows.net",
                "index": "payroll",
                "use_managed_identity": True,
            },
            "schema": {"mode": "observed"},
            "output_prefix": "rag_",
            "query_field": "text",
        }

    def test_helper_rejects_persist_directory_outside_allowed(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_path

        error = _validate_transform_provider_config_path(
            {"provider": "chroma", "provider_config": {"persist_directory": "/tmp/elspeth-outside"}},
            data_dir="/data",
        )
        assert error is not None
        assert "persist_directory" in error

    def test_helper_allows_persist_directory_under_outputs(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_path

        error = _validate_transform_provider_config_path(
            {"provider": "chroma", "provider_config": {"persist_directory": "/data/outputs/chroma"}},
            data_dir="/data",
        )
        assert error is None

    def test_helper_skips_non_rag_transform(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_path

        error = _validate_transform_provider_config_path({"some_field": "value"}, data_dir="/data")
        assert error is None

    def test_helper_skips_null_persist_directory(self) -> None:
        # A null nested path value must be skipped cleanly (no TypeError from
        # Path(None)) — parity with the runtime siblings (service/validation),
        # which guard with ``value is not None`` before resolving.
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_path

        error = _validate_transform_provider_config_path(
            {"provider": "chroma", "provider_config": {"persist_directory": None, "collection": "docs"}},
            data_dir="/data",
        )
        assert error is None

    def test_helper_rejects_azure_search_managed_identity(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_policy

        error = _validate_transform_provider_config_policy(self._azure_search_managed_identity_options())
        assert error is not None
        assert "managed identity" in error.lower()

    def test_helper_rejects_string_true_managed_identity(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_policy

        options = self._azure_search_managed_identity_options()
        options["provider_config"]["use_managed_identity"] = "true"

        error = _validate_transform_provider_config_policy(options)
        assert error is not None
        assert "managed identity" in error.lower()

    def test_upsert_node_rejects_azure_search_managed_identity(self) -> None:
        state = _empty_state()
        catalog = self._catalog_with_rag()
        result = execute_tool(
            "upsert_node",
            {
                "id": "rag",
                "node_type": "transform",
                "plugin": "rag_retrieval",
                "input": "rows",
                "on_success": "retrieved",
                "on_error": "discard",
                "options": self._azure_search_managed_identity_options(),
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False
        assert "managed identity" in result.data["error"].lower()

    def test_patch_node_options_rejects_azure_search_managed_identity(self) -> None:
        state = _empty_state()
        catalog = self._catalog_with_rag()
        created = execute_tool(
            "upsert_node",
            {
                "id": "rag",
                "node_type": "transform",
                "plugin": "rag_retrieval",
                "input": "rows",
                "on_success": "retrieved",
                "on_error": "discard",
                "options": self._rag_options("/data/outputs/chroma"),
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert created.success is True

        result = execute_tool(
            "patch_node_options",
            {
                "node_id": "rag",
                "patch": {
                    "provider": "azure_search",
                    "provider_config": {
                        "endpoint": "https://tenant-b.search.windows.net",
                        "index": "payroll",
                        "use_managed_identity": True,
                    },
                },
            },
            created.updated_state,
            catalog,
            data_dir="/data",
        )

        assert result.success is False
        assert "managed identity" in result.data["error"].lower()

    def test_set_pipeline_rejects_azure_search_managed_identity(self) -> None:
        state = _empty_state()
        catalog = self._catalog_with_rag()
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "source_out",
                "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            "nodes": [
                {
                    "id": "rag",
                    "node_type": "transform",
                    "plugin": "rag_retrieval",
                    "input": "source_out",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": self._azure_search_managed_identity_options(),
                }
            ],
            "edges": [{"id": "e1", "from_node": "source", "to_node": "rag", "edge_type": "on_success", "label": None}],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "csv",
                    "options": {
                        "path": "/data/outputs/out.csv",
                        "schema": {"mode": "observed"},
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
        }
        result = execute_tool("set_pipeline", args, state, catalog, data_dir="/data")
        assert result.success is False
        assert "managed identity" in result.data["error"].lower()

    def test_helper_allows_azure_search_api_key(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_policy

        options = self._azure_search_managed_identity_options()
        options["provider_config"] = {
            "endpoint": "https://tenant-a.search.windows.net",
            "index": "docs",
            "api_key": "test-key",
        }
        assert _validate_transform_provider_config_policy(options) is None

    def test_upsert_node_rejects_persist_directory_outside_allowed(self) -> None:
        state = _empty_state()
        catalog = self._catalog_with_rag()
        result = execute_tool(
            "upsert_node",
            {
                "id": "rag",
                "node_type": "transform",
                "plugin": "rag_retrieval",
                "input": "rows",
                "on_success": "retrieved",
                "on_error": "discard",
                "options": self._rag_options("/etc/cron.d/backdoor"),
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False
        assert "persist_directory" in result.data["error"]

    def test_upsert_node_accepts_persist_directory_under_outputs(self) -> None:
        state = _empty_state()
        catalog = self._catalog_with_rag()
        result = execute_tool(
            "upsert_node",
            {
                "id": "rag",
                "node_type": "transform",
                "plugin": "rag_retrieval",
                "input": "rows",
                "on_success": "retrieved",
                "on_error": "discard",
                "options": self._rag_options("/data/outputs/chroma"),
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is True

    def test_patch_node_options_rejects_persist_directory_outside_allowed(self) -> None:
        state = _empty_state()
        catalog = self._catalog_with_rag()
        created = execute_tool(
            "upsert_node",
            {
                "id": "rag",
                "node_type": "transform",
                "plugin": "rag_retrieval",
                "input": "rows",
                "on_success": "retrieved",
                "on_error": "discard",
                "options": self._rag_options("/data/outputs/chroma"),
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert created.success is True
        result = execute_tool(
            "patch_node_options",
            {
                "node_id": "rag",
                "patch": {"provider_config": {"persist_directory": "/etc/passwd", "collection": "docs"}},
            },
            created.updated_state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False
        assert "persist_directory" in result.data["error"]

    def test_set_pipeline_rejects_persist_directory_outside_allowed(self) -> None:
        """Parity: a bulk set_pipeline must reject an escaping transform path,
        not just escaping sink paths."""
        state = _empty_state()
        catalog = self._catalog_with_rag()
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "source_out",
                "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            "nodes": [
                {
                    "id": "rag",
                    "node_type": "transform",
                    "plugin": "rag_retrieval",
                    "input": "source_out",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": self._rag_options("/etc/cron.d/backdoor"),
                }
            ],
            "edges": [{"id": "e1", "from_node": "source", "to_node": "rag", "edge_type": "on_success", "label": None}],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "csv",
                    "options": {"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    "on_write_failure": "discard",
                }
            ],
        }
        result = execute_tool("set_pipeline", args, state, catalog, data_dir="/data")
        assert result.success is False
        assert "persist_directory" in result.data["error"]


class TestTransformLlmRetryBudgetPolicy:
    """Composer must not persist unsafe sequential multi-query LLM retry budgets."""

    @staticmethod
    def _catalog_with_llm() -> MagicMock:
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="llm",
                description="LLM transform",
                plugin_type="transform",
                config_fields=[],
            ),
        ]
        return catalog

    @staticmethod
    def _llm_multi_query_options(**overrides: Any) -> dict[str, Any]:
        options = _llm_options_with_api_key({"secret_ref": "OPENROUTER_API_KEY"})
        options.update(
            {
                "prompt_template": "Classify {{ text }}.",
                "required_input_fields": [],
                "queries": [
                    {
                        "name": "classify",
                        "input_fields": {"text": "body"},
                    }
                ],
            }
        )
        options.update(overrides)
        return options

    def test_helper_rejects_default_sequential_multi_query_retry_budget(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_policy

        error = _validate_transform_provider_config_policy(
            self._llm_multi_query_options(),
            plugin="llm",
        )
        assert error is not None
        assert "sequential multi-query llm" in error.lower()

    def test_helper_allows_small_sequential_multi_query_retry_budget(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_policy

        error = _validate_transform_provider_config_policy(
            self._llm_multi_query_options(max_capacity_retry_seconds=30.0),
            plugin="llm",
        )
        assert error is None

    def test_helper_allows_pooled_multi_query_default_retry_budget(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_policy

        error = _validate_transform_provider_config_policy(
            self._llm_multi_query_options(pool_size="2.0"),
            plugin="llm",
        )
        assert error is None

    def test_helper_rejects_fractional_multi_query_pool_size(self) -> None:
        from elspeth.web.composer.tools._common import _validate_transform_provider_config_policy

        error = _validate_transform_provider_config_policy(
            self._llm_multi_query_options(pool_size="2.5"),
            plugin="llm",
        )
        assert error is not None
        assert "sequential multi-query llm" in error.lower()

    def test_upsert_node_rejects_default_sequential_multi_query_retry_budget(self) -> None:
        result = execute_tool(
            "upsert_node",
            {
                "id": "llm_review",
                "node_type": "transform",
                "plugin": "llm",
                "input": "rows",
                "on_success": "reviewed",
                "on_error": "discard",
                "options": self._llm_multi_query_options(),
            },
            _empty_state(),
            self._catalog_with_llm(),
            data_dir="/data",
        )
        assert result.success is False
        assert "sequential multi-query llm" in result.data["error"].lower()

    def test_patch_node_options_rejects_oversized_sequential_multi_query_retry_budget(self) -> None:
        catalog = self._catalog_with_llm()
        created = execute_tool(
            "upsert_node",
            {
                "id": "llm_review",
                "node_type": "transform",
                "plugin": "llm",
                "input": "rows",
                "on_success": "reviewed",
                "on_error": "discard",
                "options": self._llm_multi_query_options(max_capacity_retry_seconds=30),
            },
            _empty_state(),
            catalog,
            data_dir="/data",
        )
        assert created.success is True

        result = execute_tool(
            "patch_node_options",
            {
                "node_id": "llm_review",
                "patch": {"max_capacity_retry_seconds": 31},
            },
            created.updated_state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False
        assert "sequential multi-query llm" in result.data["error"].lower()

    def test_set_pipeline_rejects_default_sequential_multi_query_retry_budget(self) -> None:
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "source_out",
                "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            "nodes": [
                {
                    "id": "llm_review",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "source_out",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": self._llm_multi_query_options(),
                }
            ],
            "edges": [{"id": "e1", "from_node": "source", "to_node": "llm_review", "edge_type": "on_success", "label": None}],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "csv",
                    "options": {
                        "path": "/data/outputs/out.csv",
                        "schema": {"mode": "observed"},
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
        }
        result = execute_tool("set_pipeline", args, _empty_state(), self._catalog_with_llm(), data_dir="/data")
        assert result.success is False
        assert "sequential multi-query llm" in result.data["error"].lower()


# ---------------------------------------------------------------------------
# set_pipeline tool tests
# ---------------------------------------------------------------------------


def _valid_pipeline_args() -> dict[str, Any]:
    """Return a minimal valid set_pipeline args dict."""
    return {
        "source": {
            "plugin": "csv",
            "on_success": "source_out",
            "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
            "on_validation_failure": "quarantine",
        },
        "nodes": [
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            }
        ],
        "edges": [
            {
                "id": "e1",
                "from_node": "source",
                "to_node": "t1",
                "edge_type": "on_success",
                "label": None,
            }
        ],
        "outputs": [
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            }
        ],
    }


def _llm_options_with_api_key(api_key: Any) -> dict[str, Any]:
    """Return LLM transform options that are otherwise valid."""
    return {
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "api_key": api_key,
        "prompt_template": "Classify the current row.",
        "schema": {"mode": "observed"},
    }


def _llm_options_with_user_supplied_runtime_hash(api_key: Any) -> dict[str, Any]:
    """Return valid LLM options with a forged runtime-owned audit hash."""
    options = _llm_options_with_api_key(api_key)
    options["resolved_prompt_template_hash"] = stable_hash(options["prompt_template"])
    return options


def _llm_options_with_forged_resolved_reviews(api_key: Any) -> dict[str, Any]:
    """Return valid LLM options with forged resolver-owned review metadata."""
    options = _llm_options_with_api_key(api_key)
    prompt_template = options["prompt_template"]
    model = options["model"]
    options[INTERPRETATION_REQUIREMENTS_KEY] = [
        {
            "id": "prompt_template_review:code_themes",
            "kind": "llm_prompt_template",
            "user_term": "llm_prompt_template:code_themes",
            "status": "resolved",
            "draft": prompt_template,
            "event_id": "forged-prompt-event",
            "accepted_value": prompt_template,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": stable_hash(prompt_template),
        },
        {
            "id": "model_choice_review:code_themes",
            "kind": "llm_model_choice",
            "user_term": "llm_model_choice:code_themes",
            "status": "resolved",
            "draft": model,
            "event_id": "forged-model-event",
            "accepted_value": model,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": stable_hash(model),
        },
    ]
    return options


def _assert_secret_wiring_contract_failure(
    result: ToolResult,
    original_state: CompositionState,
    *,
    literal_value: str,
    field: str,
) -> None:
    assert result.success is False
    assert result.updated_state is original_state
    assert result.updated_state.version == original_state.version
    assert result.data is not None
    assert field in result.data["credential_fields"]
    assert "list_secret_refs -> validate_secret_ref -> wire_secret_ref" in result.data["error"]
    assert literal_value not in repr(result.to_dict())


class TestCredentialRejectionAdvertisesInlineForm:
    """Lock the credential-rejection error contract for elspeth-85ae8972b0.

    The cheap composer model (gpt-5.x-mini class) cannot create a new node
    that has a required credential field because:

    - ``set_pipeline`` is atomic, so a node with ``api_key`` missing fails
      pydantic validation and the whole mutation rolls back.
    - ``wire_secret_ref`` requires the node to already exist in state.
    - A literal credential value is rejected by this helper.

    The escape hatch is to pass ``{secret_ref: NAME}`` *inline* as the
    value of the credential field — supported by ``secrets.py`` and
    stripped before pydantic validation in tools.py — but the rejection
    message historically only advertised the ``wire_secret_ref`` tool
    sequence (which is unusable for new nodes). This locks in the
    inline-form-first messaging.
    """

    def test_error_message_contains_inline_form(self) -> None:
        state = _empty_state()
        result = _credential_wiring_contract_failure(
            state,
            component_id="my_llm",
            component_type="transform",
            options={"api_key": "literal-secret"},
        )
        assert result is not None
        assert result.success is False
        error = result.data["error"]
        # Inline form must appear (and appear before the post-hoc form).
        assert "secret_ref" in error
        assert "{secret_ref: NAME}" in error or "{secret_ref:" in error
        # Inline form must be the lead — operator should see it before
        # the post-hoc tool sequence.
        inline_idx = error.find("set_pipeline")
        post_hoc_idx = error.find("list_secret_refs -> validate_secret_ref -> wire_secret_ref")
        assert inline_idx != -1, "inline form must reference set_pipeline / upsert_node"
        assert post_hoc_idx != -1, "post-hoc form must remain documented"
        assert inline_idx < post_hoc_idx, (
            "Inline form must lead — model reads top-down and the wire_secret_ref path is unusable for new nodes."
        )

    def test_repair_payload_splits_inline_and_post_hoc(self) -> None:
        state = _empty_state()
        result = _credential_wiring_contract_failure(
            state,
            component_id="my_llm",
            component_type="transform",
            options={"api_key": "literal-secret"},
        )
        assert result is not None
        repair = result.data["repair"]
        # Old key must be gone — no compatibility shim per CLAUDE.md.
        assert "required_tool_sequence" not in repair
        # New shape: two separately-keyed forms.
        assert "inline_form" in repair
        assert "post_hoc_form" in repair
        inline = repair["inline_form"]
        post_hoc = repair["post_hoc_form"]
        # Inline form carries an example the model can copy verbatim.
        assert "instruction" in inline
        assert "example_options" in inline
        assert inline["example_options"] == {"api_key": {"secret_ref": "<NAME>"}}
        # Post-hoc form carries the tool sequence.
        assert "instruction" in post_hoc
        assert tuple(post_hoc["tool_sequence"]) == (
            "list_secret_refs",
            "validate_secret_ref",
            "wire_secret_ref",
        )

    def test_existing_helper_assertion_still_holds(self) -> None:
        """The legacy ``_assert_secret_wiring_contract_failure`` helper
        looks for the post-hoc tool-sequence substring — keep it intact
        so existing rejection-shape tests continue to lock in the
        contract from the post-hoc side."""
        state = _empty_state()
        result = _credential_wiring_contract_failure(
            state,
            component_id="my_llm",
            component_type="transform",
            options={"api_key": "literal-secret"},
        )
        assert result is not None
        assert "list_secret_refs -> validate_secret_ref -> wire_secret_ref" in result.data["error"]

    def test_multiple_fields_listed_with_single_inline_example_form(self) -> None:
        """When multiple credential fields are violated, the example_options
        payload should enumerate all of them so the model sees the inline
        form for each one."""
        state = _empty_state()
        result = _credential_wiring_contract_failure(
            state,
            component_id="db",
            component_type="sink",
            options={"api_key": "literal-1", "password": "literal-2"},
        )
        assert result is not None
        example = result.data["repair"]["inline_form"]["example_options"]
        assert set(example.keys()) == {"api_key", "password"}
        assert all(v == {"secret_ref": "<NAME>"} for v in example.values())


class TestSetPipelineSchemaShape:
    """Lock the ``set_pipeline`` schema shape so the elspeth-4e79436719 Bug A
    regression cannot return via a future schema edit.

    The walker treats nested ``required`` as conditional-on-presence; for that
    contract to hold, ``inline_blob`` must remain *outside* the outer
    ``source.required`` list. If a future commit adds ``inline_blob`` to the
    outer ``required`` list, every regular ``set_pipeline`` call (which
    correctly does not ship inline literal data) starts failing pre-dispatch.
    """

    def test_inline_blob_is_optional_at_source_level(self) -> None:
        from elspeth.web.composer.tools import get_tool_definitions

        defns = {d["name"]: d for d in get_tool_definitions()}
        source = defns["set_pipeline"]["parameters"]["properties"]["source"]
        assert "inline_blob" in source["properties"], "inline_blob property must exist on source"
        assert "inline_blob" not in source["required"], (
            "inline_blob must remain optional at source level — adding it to "
            "source.required reintroduces elspeth-4e79436719 Bug A by forcing "
            "every regular set_pipeline call to ship literal inline content."
        )

    def test_inline_blob_inner_required_is_well_formed(self) -> None:
        from elspeth.web.composer.tools import get_tool_definitions

        defns = {d["name"]: d for d in get_tool_definitions()}
        inline = defns["set_pipeline"]["parameters"]["properties"]["source"]["properties"]["inline_blob"]
        # The conditional-on-presence semantics rely on this list being non-empty
        # — if inline_blob is supplied, these are the fields the walker enforces.
        assert set(inline["required"]) == {"filename", "mime_type", "content"}

    def test_existing_blob_binding_is_optional_at_source_level(self) -> None:
        """Atomic set_pipeline must expose an existing-blob binding path."""
        from elspeth.web.composer.tools import get_tool_definitions

        defns = {d["name"]: d for d in get_tool_definitions()}
        source = defns["set_pipeline"]["parameters"]["properties"]["source"]
        assert "blob_id" in source["properties"], "source.blob_id must bind an already uploaded blob"
        assert "blob_id" not in source["required"]

    def test_options_are_optional_at_schema_boundary(self) -> None:
        """Missing options must reach the handler so it can emit repair guidance."""
        from elspeth.web.composer.tools import get_tool_definitions

        defns = {d["name"]: d for d in get_tool_definitions()}
        params = defns["set_pipeline"]["parameters"]
        source = params["properties"]["source"]
        output_item = params["properties"]["outputs"]["items"]

        assert "options" not in source["required"]
        assert "options" not in output_item["required"]

    def test_output_schema_examples_show_valid_file_sink_options(self) -> None:
        from elspeth.web.composer.tools import get_tool_definitions

        defns = {d["name"]: d for d in get_tool_definitions()}
        output_item = defns["set_pipeline"]["parameters"]["properties"]["outputs"]["items"]

        assert {
            "sink_name": "results",
            "plugin": "json",
            "options": {
                "path": "outputs/results.json",
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
            },
            "on_write_failure": "discard",
        } in output_item["examples"]


class TestSetPipeline:
    def _catalog_with_json_sink(self) -> MagicMock:
        catalog = _mock_catalog()
        catalog.list_sinks.return_value = [
            *catalog.list_sinks.return_value,
            PluginSummary(
                name="json",
                description="JSON file sink",
                plugin_type="sink",
                config_fields=[],
            ),
        ]
        return catalog

    def test_set_pipeline_creates_valid_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("set_pipeline", _valid_pipeline_args(), state, catalog)
        assert result.success is True
        assert result.validation is not None
        assert result.validation.is_valid is True
        assert result.updated_state.version == 2  # incremented from 1

    def test_set_pipeline_accepts_named_sources_mapping(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args.pop("source")
        args["sources"] = {
            "customers": {
                "plugin": "csv",
                "on_success": "customer_rows",
                "options": {"path": "/data/customers.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "orders": {
                "plugin": "json",
                "on_success": "order_rows",
                "options": {"path": "/data/orders.json", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
        }
        args["nodes"] = []
        args["edges"] = []
        args["outputs"] = [
            {
                "sink_name": "customer_rows",
                "plugin": "csv",
                "options": {"path": "/data/customer_rows.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            {
                "sink_name": "order_rows",
                "plugin": "json",
                "options": {"path": "/data/order_rows.json", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
        ]

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is True
        assert tuple(result.updated_state.sources) == ("customers", "orders")
        assert result.updated_state.sources["customers"].plugin == "csv"
        assert "source:customers" in result.affected_nodes
        assert "source:orders" in result.affected_nodes

    def test_set_pipeline_source_defaults_validation_failures_to_discard(self) -> None:
        """Omitting on_validation_failure must not synthesize an absent quarantine sink."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        del args["source"]["on_validation_failure"]

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is True
        assert _default_source(result.updated_state) is not None
        assert _default_source(result.updated_state).on_validation_failure == "discard"
        assert result.data is None

    def test_set_pipeline_missing_json_output_options_returns_exact_repair_hint(self) -> None:
        state = _empty_state()
        catalog = self._catalog_with_json_sink()
        args = _valid_pipeline_args()
        args["source"]["options"]["path"] = "/data/blobs/in.csv"
        del args["outputs"][0]["options"]
        args["outputs"][0]["plugin"] = "json"

        result = execute_tool("set_pipeline", args, state, catalog, data_dir="/data")

        assert result.success is False
        assert result.updated_state is state
        error = result.data["error"]
        assert "Output 'main' is missing options" in error
        assert '"sink_name": "main"' in error
        assert '"plugin": "json"' in error
        assert '"path": "outputs/main.json"' in error
        assert '"schema": {"mode": "observed"}' in error
        assert '"collision_policy": "auto_increment"' in error
        assert '"on_write_failure": "discard"' in error

    def test_set_pipeline_failure_leads_validation_with_rejection_reason(self) -> None:
        """Regression for composer session 58d7ede3 round 6.

        When ``set_pipeline`` rejects a mutation, ``validation.errors[0]``
        must carry the actionable rejection reason (component
        ``rejected_mutation``) ahead of any state-snapshot errors like
        ``"No source configured."``. In the live session, the LLM read
        the stale-state errors first and burned a full round retrying
        with only a cosmetic change.
        """
        state = _empty_state()
        catalog = self._catalog_with_json_sink()
        args = _valid_pipeline_args()
        args["source"]["options"]["path"] = "/data/blobs/in.csv"
        del args["outputs"][0]["options"]
        args["outputs"][0]["plugin"] = "json"

        result = execute_tool("set_pipeline", args, state, catalog, data_dir="/data")

        assert result.success is False
        first = result.validation.errors[0]
        assert first.component == "rejected_mutation"
        assert first.severity == "high"
        assert "missing options" in first.message.lower()
        # data.error mirrors the leading entry's message verbatim so the
        # two channels stay in sync.
        assert first.message == result.data["error"]
        # Stale state-level errors remain in the array — they must not
        # vanish, only be demoted from the leading slot.
        components = [e.component for e in result.validation.errors[1:]]
        assert "source" in components
        assert "pipeline" in components

    def test_set_pipeline_accepts_two_json_sinks_with_explicit_file_options(self, tmp_path: Path) -> None:
        state = _empty_state()
        catalog = self._catalog_with_json_sink()
        args = _valid_pipeline_args()
        args["source"]["options"]["path"] = str(tmp_path / "blobs" / "input.csv")
        args["nodes"][0]["on_error"] = "failures"
        args["outputs"] = [
            {
                "sink_name": "main",
                "plugin": "json",
                "options": {
                    "path": str(tmp_path / "outputs" / "main.json"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            {
                "sink_name": "failures",
                "plugin": "json",
                "options": {
                    "path": str(tmp_path / "outputs" / "failures.json"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
        ]

        result = execute_tool("set_pipeline", args, state, catalog, data_dir=str(tmp_path))

        assert result.success is True
        assert result.validation.is_valid is True
        assert [output.plugin for output in result.updated_state.outputs] == ["json", "json"]
        assert [output.name for output in result.updated_state.outputs] == ["main", "failures"]

    def test_set_pipeline_rejects_literal_credential_value_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        literal = "literal-api-key-for-test"
        args = _valid_pipeline_args()
        args["nodes"][0] = {
            "id": "code_themes",
            "node_type": "transform",
            "plugin": "llm",
            "input": "source_out",
            "on_success": "main",
            "on_error": "discard",
            "options": _llm_options_with_api_key(literal),
        }
        args["edges"][0]["to_node"] = "code_themes"

        result = execute_tool("set_pipeline", args, state, catalog)

        _assert_secret_wiring_contract_failure(
            result,
            state,
            literal_value=literal,
            field="code_themes:api_key",
        )

    def test_set_pipeline_rejects_placeholder_database_table_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.list_sinks.return_value = [
            *catalog.list_sinks.return_value,
            PluginSummary(
                name="database",
                description="Database sink",
                plugin_type="sink",
                config_fields=[],
            ),
        ]
        args = _valid_pipeline_args()
        args["outputs"][0] = {
            "sink_name": "main",
            "plugin": "database",
            "options": {
                "url": {"secret_ref": "DATABASE_URL"},
                "table": "<OPERATOR_REQUIRED>",
                "schema": {"mode": "observed"},
            },
            "on_write_failure": "discard",
        }

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is False
        assert result.updated_state is state
        assert result.data is not None
        error = result.data["error"]
        assert "Output 'main'" in error
        assert "table" in error
        assert "placeholder" in error

    def test_set_pipeline_accepts_wired_secret_ref_marker(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"][0] = {
            "id": "code_themes",
            "node_type": "transform",
            "plugin": "llm",
            "input": "source_out",
            "on_success": "main",
            "on_error": "discard",
            "options": _llm_options_with_api_key({"secret_ref": "OPENROUTER_API_KEY"}),
        }
        args["edges"][0]["to_node"] = "code_themes"

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is True
        node = result.updated_state.nodes[0]
        assert node.options["api_key"] == {"secret_ref": "OPENROUTER_API_KEY"}

    def test_set_pipeline_rejects_user_supplied_llm_runtime_hash_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"][0] = {
            "id": "code_themes",
            "node_type": "transform",
            "plugin": "llm",
            "input": "source_out",
            "on_success": "main",
            "on_error": "discard",
            "options": _llm_options_with_user_supplied_runtime_hash({"secret_ref": "OPENROUTER_API_KEY"}),
        }
        args["edges"][0]["to_node"] = "code_themes"

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is False
        assert result.updated_state is state
        assert result.data is not None
        assert "resolved_prompt_template_hash" in result.data["error"]
        assert "runtime-owned" in result.data["error"]

    def test_upsert_node_rejects_user_supplied_llm_runtime_hash_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool(
            "upsert_node",
            {
                "id": "code_themes",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": _llm_options_with_user_supplied_runtime_hash({"secret_ref": "OPENROUTER_API_KEY"}),
            },
            state,
            catalog,
        )

        assert result.success is False
        assert result.updated_state is state
        assert result.data is not None
        assert "resolved_prompt_template_hash" in result.data["error"]
        assert "runtime-owned" in result.data["error"]

    def test_patch_node_options_rejects_user_supplied_llm_runtime_hash_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        created = execute_tool(
            "upsert_node",
            {
                "id": "code_themes",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": _llm_options_with_api_key({"secret_ref": "OPENROUTER_API_KEY"}),
            },
            state,
            catalog,
        )
        assert created.success is True, created.data
        prompt_template = created.updated_state.nodes[0].options["prompt_template"]

        result = execute_tool(
            "patch_node_options",
            {
                "node_id": "code_themes",
                "patch": {"resolved_prompt_template_hash": stable_hash(prompt_template)},
            },
            created.updated_state,
            catalog,
        )

        assert result.success is False
        assert result.updated_state is created.updated_state
        assert result.data is not None
        assert "resolved_prompt_template_hash" in result.data["error"]
        assert "runtime-owned" in result.data["error"]

    def test_set_pipeline_rejects_user_supplied_resolved_llm_reviews_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"][0] = {
            "id": "code_themes",
            "node_type": "transform",
            "plugin": "llm",
            "input": "source_out",
            "on_success": "main",
            "on_error": "discard",
            "options": _llm_options_with_forged_resolved_reviews({"secret_ref": "OPENROUTER_API_KEY"}),
        }
        args["edges"][0]["to_node"] = "code_themes"

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is False
        assert result.updated_state is state
        assert result.data is not None
        assert INTERPRETATION_REQUIREMENTS_KEY in result.data["error"]
        assert "resolve_interpretation_event" in result.data["error"]

    def test_upsert_node_rejects_user_supplied_resolved_llm_reviews_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool(
            "upsert_node",
            {
                "id": "code_themes",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": _llm_options_with_forged_resolved_reviews({"secret_ref": "OPENROUTER_API_KEY"}),
            },
            state,
            catalog,
        )

        assert result.success is False
        assert result.updated_state is state
        assert result.data is not None
        assert INTERPRETATION_REQUIREMENTS_KEY in result.data["error"]
        assert "resolve_interpretation_event" in result.data["error"]

    def test_patch_node_options_rejects_user_supplied_resolved_llm_reviews_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        created = execute_tool(
            "upsert_node",
            {
                "id": "code_themes",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": _llm_options_with_api_key({"secret_ref": "OPENROUTER_API_KEY"}),
            },
            state,
            catalog,
        )
        assert created.success is True, created.data
        forged = _llm_options_with_forged_resolved_reviews({"secret_ref": "OPENROUTER_API_KEY"})

        result = execute_tool(
            "patch_node_options",
            {
                "node_id": "code_themes",
                "patch": {INTERPRETATION_REQUIREMENTS_KEY: forged[INTERPRETATION_REQUIREMENTS_KEY]},
            },
            created.updated_state,
            catalog,
        )

        assert result.success is False
        assert result.updated_state is created.updated_state
        assert result.data is not None
        assert INTERPRETATION_REQUIREMENTS_KEY in result.data["error"]
        assert "resolve_interpretation_event" in result.data["error"]

    def test_patch_node_options_preserves_existing_resolved_llm_reviews_on_unrelated_patch(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        created = execute_tool(
            "upsert_node",
            {
                "id": "code_themes",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": _llm_options_with_api_key({"secret_ref": "OPENROUTER_API_KEY"}),
            },
            state,
            catalog,
        )
        assert created.success is True, created.data
        resolved_options = _llm_options_with_forged_resolved_reviews({"secret_ref": "OPENROUTER_API_KEY"})
        resolved_options["resolved_prompt_template_hash"] = stable_hash(resolved_options["prompt_template"])
        resolved_node = replace(created.updated_state.nodes[0], options=resolved_options)
        resolved_state = created.updated_state.with_node(resolved_node)

        result = execute_tool(
            "patch_node_options",
            {"node_id": "code_themes", "patch": {"temperature": 0.1}},
            resolved_state,
            catalog,
        )

        assert result.success is True, result.data
        patched_options = result.updated_state.nodes[0].options
        assert patched_options["temperature"] == 0.1
        assert patched_options["resolved_prompt_template_hash"] == stable_hash(patched_options["prompt_template"])
        assert deep_thaw(patched_options[INTERPRETATION_REQUIREMENTS_KEY]) == resolved_options[INTERPRETATION_REQUIREMENTS_KEY]

    def test_set_pipeline_rejects_secret_ref_in_non_credential_field(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"][0] = {
            "id": "scrape_pages",
            "node_type": "transform",
            "plugin": "web_scrape",
            "input": "source_out",
            "on_success": "main",
            "on_error": "discard",
            "options": {
                "url_field": "url",
                "http": {
                    "abuse_contact": {"secret_ref": "ANY_SECRET"},
                    "scraping_reason": "research",
                    "allowed_hosts": ["example.com"],
                },
            },
        }
        args["edges"][0]["to_node"] = "scrape_pages"

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is False
        assert result.updated_state is state
        assert "web_scrape" in result.data["error"]
        assert "http.abuse_contact" in result.data["error"]
        assert "ANY_SECRET" in result.data["error"]
        assert "only credential-bearing fields" in result.data["error"]
        assert "api_key" in result.data["error"]

    def test_upsert_node_rejects_secret_ref_in_non_credential_field(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool(
            "upsert_node",
            {
                "id": "scrape_pages",
                "node_type": "transform",
                "plugin": "web_scrape",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": {
                    "url_field": "url",
                    "http": {
                        "scraping_reason": {"secret_ref": "ANY_SECRET"},
                        "abuse_contact": "ops@example.com",
                        "allowed_hosts": ["example.com"],
                    },
                },
            },
            state,
            catalog,
        )

        assert result.success is False
        assert result.updated_state is state
        assert "web_scrape" in result.data["error"]
        assert "http.scraping_reason" in result.data["error"]
        assert "ANY_SECRET" in result.data["error"]
        assert "only credential-bearing fields" in result.data["error"]

    def test_upsert_node_rejects_literal_credential_value_without_mutating_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        literal = "literal-upsert-key-for-test"

        result = execute_tool(
            "upsert_node",
            {
                "id": "code_themes",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": _llm_options_with_api_key(literal),
            },
            state,
            catalog,
        )

        _assert_secret_wiring_contract_failure(
            result,
            state,
            literal_value=literal,
            field="code_themes:api_key",
        )

    def test_upsert_node_llm_prevalidation_ignores_review_metadata(self) -> None:
        """LLM plugin validation strips web-only prompt review metadata but keeps it in state."""
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool(
            "upsert_node",
            {
                "id": "code_themes",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    "prompt_template": "Classify pending interpretation: {{ row.text }}",
                    "schema": {"mode": "observed"},
                    PROMPT_TEMPLATE_PARTS_KEY: [
                        {"kind": "text", "text": "Classify "},
                        {"kind": "interpretation_ref", "requirement_id": "prompt_review"},
                        {"kind": "text", "text": ": {{ row.text }}"},
                    ],
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "prompt_review",
                            "kind": "llm_prompt_template",
                            "user_term": "llm_prompt_template:code_themes",
                            "status": "pending",
                            "draft": "Classify pending interpretation: {{ row.text }}",
                            "event_id": None,
                            "accepted_value": None,
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": None,
                        }
                    ],
                },
            },
            state,
            catalog,
        )

        assert result.success is True, result.data
        node = result.updated_state.nodes[0]
        assert PROMPT_TEMPLATE_PARTS_KEY in node.options
        assert INTERPRETATION_REQUIREMENTS_KEY in node.options

    def test_set_pipeline_rejects_manual_blob_ref_in_source_options(self) -> None:
        """Manual blob_ref must not bypass the blob-backed source binding tools."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["source"]["options"] = {
            "blob_ref": "manual-ref",
            "path": "/tmp/not_the_blob_storage.csv",
            "schema": {"mode": "observed"},
        }

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is False
        assert _default_source(result.updated_state) is None
        assert result.updated_state.version == 1
        assert "set_source_from_blob" in result.data["error"]
        assert "source.inline_blob" in result.data["error"]

    def test_set_pipeline_binds_existing_blob_instead_of_header_only_sibling(self, tmp_path: Path) -> None:
        """Complete-pipeline writes must preserve the user-selected uploaded blob."""
        from datetime import UTC, datetime

        state = _empty_state()
        catalog = _mock_catalog()
        engine, session_id = _session_engine_with_session()
        uploaded_id = str(uuid4())
        header_only_id = str(uuid4())
        storage_dir = tmp_path / "blobs" / session_id
        storage_dir.mkdir(parents=True)
        uploaded_path = storage_dir / f"{uploaded_id}_contact_form_submissions.csv"
        header_only_path = storage_dir / f"{header_only_id}_hubspot_export.csv"
        uploaded_content = (
            "submission_id,timestamp,name,email,company,message\n"
            "CF-501,2026-08-01T09:14Z,Sarah Patel,sarah@example.com,Northwind,Interested in pricing\n"
            "CF-502,2026-08-01T09:22Z,Bot Bot,spam@example.test,,CHEAP SEO SERVICES\n"
        )
        header_only_content = "submission_id,timestamp,name,email,company,message\n"
        uploaded_path.write_text(uploaded_content, encoding="utf-8")
        header_only_path.write_text(header_only_content, encoding="utf-8")
        now = datetime.now(UTC)
        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert(),
                [
                    {
                        "id": uploaded_id,
                        "session_id": session_id,
                        "filename": "contact_form_submissions.csv",
                        "mime_type": "text/csv",
                        "size_bytes": len(uploaded_content.encode("utf-8")),
                        "content_hash": _STUB_SHA256,
                        "storage_path": str(uploaded_path),
                        "created_at": now,
                        "created_by": "user",
                        "source_description": "scenario uploaded contact form submissions",
                        "status": "ready",
                    },
                    {
                        "id": header_only_id,
                        "session_id": session_id,
                        "filename": "hubspot_export.csv",
                        "mime_type": "text/csv",
                        "size_bytes": len(header_only_content.encode("utf-8")),
                        "content_hash": _STUB_SHA256_ALT,
                        "storage_path": str(header_only_path),
                        "created_at": now,
                        "created_by": "assistant",
                        "source_description": "header-only inferred schema",
                        "status": "ready",
                    },
                ],
            )

        args = _valid_pipeline_args()
        args["source"] = {
            "plugin": "csv",
            "blob_id": uploaded_id,
            "on_success": "source_out",
            "options": {"schema": {"mode": "observed"}},
            "on_validation_failure": "quarantine",
        }
        args["outputs"][0]["options"]["path"] = str(tmp_path / "outputs" / "out.csv")
        args["outputs"][0]["options"]["mode"] = "write"
        args["outputs"][0]["options"]["collision_policy"] = "auto_increment"

        result = execute_tool(
            "set_pipeline",
            args,
            state,
            catalog,
            data_dir=str(tmp_path),
            session_engine=engine,
            session_id=session_id,
        )

        assert result.success is True
        assert _default_source(result.updated_state) is not None
        source_options = _default_source(result.updated_state).options
        assert source_options["blob_ref"] == uploaded_id
        assert source_options["path"] == str(uploaded_path)
        assert source_options["path"] != str(header_only_path)
        assert result.data["source_blob"]["blob_id"] == uploaded_id

    def test_set_pipeline_rejects_header_only_inline_csv_when_uploaded_csv_exists(self, tmp_path: Path) -> None:
        """Header-only inline CSV must not supersede an uploaded ready CSV."""
        from datetime import UTC, datetime

        state = _empty_state()
        catalog = _mock_catalog()
        engine, session_id = _session_engine_with_session()
        uploaded_id = str(uuid4())
        uploaded_content = "name,email\nAlice,alice@example.com\n"
        uploaded_path = tmp_path / "blobs" / session_id / f"{uploaded_id}_contacts.csv"
        uploaded_path.parent.mkdir(parents=True)
        uploaded_path.write_text(uploaded_content, encoding="utf-8")
        now = datetime.now(UTC)
        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=uploaded_id,
                    session_id=session_id,
                    filename="contacts.csv",
                    mime_type="text/csv",
                    size_bytes=len(uploaded_content.encode("utf-8")),
                    content_hash=_STUB_SHA256,
                    storage_path=str(uploaded_path),
                    created_at=now,
                    created_by="user",
                    source_description="uploaded contact rows",
                    status="ready",
                )
            )

        args = _valid_pipeline_args()
        args["source"] = {
            "plugin": "csv",
            "on_success": "source_out",
            "options": {"schema": {"mode": "observed"}},
            "inline_blob": {
                "filename": "contacts.csv",
                "mime_type": "text/csv",
                "content": "name,email\n",
            },
            "on_validation_failure": "quarantine",
        }
        args["outputs"][0]["options"]["path"] = str(tmp_path / "outputs" / "out.csv")
        args["outputs"][0]["options"]["mode"] = "write"
        args["outputs"][0]["options"]["collision_policy"] = "auto_increment"

        result = execute_tool(
            "set_pipeline",
            args,
            state,
            catalog,
            data_dir=str(tmp_path),
            session_engine=engine,
            session_id=session_id,
            **_verbatim_blob_context(engine, session_id, "name,email\n"),
        )

        assert result.success is False
        assert _default_source(result.updated_state) is None
        assert "header-only inline CSV" in result.data["error"]
        assert uploaded_id in result.data["error"]

    def test_set_pipeline_unknown_source_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: nonexistent")
        args = _valid_pipeline_args()
        args["source"]["plugin"] = "nonexistent"
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is False
        assert "source" in result.data["error"].lower()

    def test_set_pipeline_unknown_node_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        from elspeth.web.catalog.schemas import PluginSchemaInfo

        def selective_schema(plugin_type: Literal["source", "transform", "sink"], name: str) -> PluginSchemaInfo:
            if plugin_type == "transform" and name == "badplugin":
                raise ValueError(f"Unknown plugin: {name}")
            return PluginSchemaInfo(
                name=name,
                plugin_type=plugin_type,
                description="",
                json_schema={},
                knob_schema={"fields": []},
            )

        catalog.get_schema.side_effect = selective_schema
        args = _valid_pipeline_args()
        args["nodes"][0]["plugin"] = "badplugin"
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is False
        assert "transform" in result.data["error"].lower()

    def test_set_pipeline_unknown_sink_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()

        from elspeth.web.catalog.schemas import PluginSchemaInfo

        def selective_schema(plugin_type: Literal["source", "transform", "sink"], name: str) -> PluginSchemaInfo:
            if plugin_type == "sink" and name == "badsink":
                raise ValueError(f"Unknown plugin: {name}")
            return PluginSchemaInfo(
                name=name,
                plugin_type=plugin_type,
                description="",
                json_schema={},
                knob_schema={"fields": []},
            )

        catalog.get_schema.side_effect = selective_schema
        args = _valid_pipeline_args()
        args["outputs"][0]["plugin"] = "badsink"
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is False
        assert "sink" in result.data["error"].lower()

    def test_set_pipeline_missing_required_field_fails(self) -> None:
        """Missing required field is now a Tier-3 ToolArgumentError.

        Post Task 14 (set_pipeline manifest promotion):
        :class:`SetPipelineArgumentsModel` / :class:`_SetPipelineSourceModel`
        require ``source.on_success``; absence raises a structured
        :class:`pydantic.ValidationError` that the handler re-raises as
        :class:`ToolArgumentError` BEFORE any handler-side spec
        construction.  Previously the omission fell through to the
        ``try: SourceSpec(...)``/``except KeyError`` branch and was
        reported via ``"Invalid pipeline spec"`` in ``result.data["error"]``;
        that branch was removed in Task 14 because Pydantic now catches
        the type errors upstream.

        The compose-loop ARG_ERROR routing at ``service.py:2480`` builds
        the LLM-facing error payload from the captured
        :class:`ToolArgumentError`.  This unit-level test reaches the
        handler directly via ``execute_tool``, so it observes the bare
        exception class — not the wrapped LLM payload.
        """
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        # Remove on_success from source — required field
        del args["source"]["on_success"]
        with pytest.raises(ToolArgumentError) as exc_info:
            execute_tool("set_pipeline", args, state, catalog)
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_set_pipeline_replaces_entire_state(self) -> None:
        # Build a state with 3 nodes first
        state = _empty_state()
        catalog = _mock_catalog()
        for i in range(3):
            r = execute_tool(
                "upsert_node",
                {
                    "id": f"t{i}",
                    "node_type": "transform",
                    "plugin": "passthrough",
                    "input": "in",
                    "on_success": "out",
                    "options": {"schema": {"mode": "observed"}},
                },
                state,
                catalog,
            )
            state = r.updated_state
        assert len(state.nodes) == 3

        # set_pipeline with 1 node replaces all
        result = execute_tool("set_pipeline", _valid_pipeline_args(), state, catalog)
        assert result.success is True
        assert len(result.updated_state.nodes) == 1
        assert result.updated_state.nodes[0].id == "t1"

    def test_set_pipeline_version_increments(self) -> None:
        from elspeth.web.composer.state import PipelineMetadata

        state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=5,
        )
        catalog = _mock_catalog()
        result = execute_tool("set_pipeline", _valid_pipeline_args(), state, catalog)
        assert result.success is True
        assert result.updated_state.version == 6

    def test_set_pipeline_validation_runs(self) -> None:
        """A pipeline with a disconnected node (unreachable input) should produce
        validation errors or is_valid=False."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        # Add a second node that has an input not connected to anything
        args["nodes"].append(
            {
                "id": "t2",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "orphan_channel",
                "on_success": "main",
                "on_error": None,
                "options": {"schema": {"mode": "observed"}},
            }
        )
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is True
        assert result.validation is not None
        # The orphan node has no reachable input — validation should flag it
        assert result.validation.is_valid is False
        assert len(result.validation.errors) > 0

    def test_set_pipeline_surfaces_fixed_schema_edge_contract_mismatch(self) -> None:
        """LLM fixed-schema requirements must reject source rows before runtime."""
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="llm",
                description="LLM transform",
                plugin_type="transform",
                config_fields=[],
            ),
        ]
        args = {
            "source": {
                "plugin": "text",
                "on_success": "rate_in",
                "options": {
                    "path": "/data/colors.txt",
                    "column": "color",
                    "schema": {"mode": "fixed", "fields": ["color: str"]},
                },
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "transform_rate_teal_pairing",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "rate_in",
                    "on_success": "results",
                    "on_error": "discard",
                    "options": {
                        "provider": "openrouter",
                        "model": "openai/gpt-4o-mini",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                        "prompt_template": "Rate {{ row.color }} for teal pairing.",
                        "schema": {
                            "mode": "fixed",
                            "fields": ["color: str", "teal_pairing_rating: str"],
                        },
                    },
                }
            ],
            "edges": [
                {
                    "id": "e1",
                    "from_node": "source",
                    "to_node": "transform_rate_teal_pairing",
                    "edge_type": "on_success",
                    "label": None,
                },
                {
                    "id": "e2",
                    "from_node": "transform_rate_teal_pairing",
                    "to_node": "results",
                    "edge_type": "on_success",
                    "label": None,
                },
            ],
            "outputs": [
                {
                    "sink_name": "results",
                    "plugin": "csv",
                    "options": {"path": "/data/results.csv", "schema": {"mode": "observed"}},
                    "on_write_failure": "discard",
                }
            ],
        }

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is True
        assert result.validation is not None
        assert result.validation.is_valid is False
        assert any("teal_pairing_rating" in error.message for error in result.validation.errors)
        contract = next(ec for ec in result.validation.edge_contracts if ec.to_id == "transform_rate_teal_pairing")
        assert contract.from_id == "source"
        assert contract.consumer_requires == ("color", "teal_pairing_rating")
        assert contract.producer_guarantees == ("color",)
        assert contract.missing_fields == ("teal_pairing_rating",)
        assert contract.satisfied is False

    def test_set_pipeline_unknown_node_type_invalidates_state(self) -> None:
        """set_pipeline keeps enum parsing recoverable but Stage 1 must reject unknown node types."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"][0]["node_type"] = "bogus"

        result = execute_tool("set_pipeline", args, state, catalog)

        assert result.success is True
        assert result.validation is not None
        assert result.validation.is_valid is False
        assert any("unknown node_type 'bogus'" in error.message for error in result.validation.errors)

    def test_set_pipeline_gate_injection_rejected(self) -> None:
        """set_pipeline rejects gate nodes with injection in condition."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"].append(
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "source_out",
                "on_success": None,
                "on_error": None,
                "options": {},
                "condition": "__import__('os').system('whoami')",
                "routes": {"true": "main", "false": "main"},
            }
        )
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is False
        assert "Forbidden construct" in result.data["error"]
        assert result.updated_state.version == 1

    def test_set_pipeline_gate_malformed_condition_rejected(self) -> None:
        """set_pipeline rejects gate nodes with syntax errors in condition."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"].append(
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "source_out",
                "on_success": None,
                "on_error": None,
                "options": {},
                "condition": "row['x'] >>>= 5",
                "routes": {"true": "main", "false": "main"},
            }
        )
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is False
        assert "Invalid gate condition syntax" in result.data["error"]

    def test_set_pipeline_gate_valid_condition_accepted(self) -> None:
        """set_pipeline accepts gate nodes with valid conditions."""
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        args["nodes"].append(
            {
                "id": "g1",
                "node_type": "gate",
                "plugin": None,
                "input": "source_out",
                "on_success": None,
                "on_error": None,
                "options": {},
                "condition": "row['score'] >= 0.5",
                "routes": {"true": "main", "false": "main"},
            }
        )
        result = execute_tool("set_pipeline", args, state, catalog)
        assert result.success is True
        gate_nodes = [n for n in result.updated_state.nodes if n.node_type == "gate"]
        assert len(gate_nodes) == 1
        assert gate_nodes[0].condition == "row['score'] >= 0.5"

    def test_set_pipeline_materializes_inline_blob_source(self, tmp_path: Path) -> None:
        """set_pipeline can atomically bind inline literal data as the source."""
        state = _empty_state()
        catalog = _mock_catalog()
        engine, session_id = _session_engine_with_session()
        output_path = tmp_path / "outputs" / "append.csv"
        args = {
            "source": {
                "plugin": "text",
                "on_success": "source_out",
                "options": {
                    "column": "text",
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                "inline_blob": {
                    "filename": "input.txt",
                    "mime_type": "text/plain",
                    "content": "hello",
                    "description": "literal input from the user prompt",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "append_world",
                    "node_type": "transform",
                    "plugin": "value_transform",
                    "input": "source_out",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": {
                        "schema": {
                            "mode": "observed",
                            "guaranteed_fields": ["text"],
                            "required_fields": ["text"],
                        },
                        "operations": [
                            {
                                "target": "text",
                                "expression": "row['text'] + ' world'",
                            }
                        ],
                    },
                }
            ],
            "edges": [
                {
                    "id": "source_to_append",
                    "from_node": "source",
                    "to_node": "append_world",
                    "edge_type": "on_success",
                },
                {
                    "id": "append_to_main",
                    "from_node": "append_world",
                    "to_node": "main",
                    "edge_type": "on_success",
                },
            ],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "csv",
                    "options": {
                        "path": str(output_path),
                        "schema": {"mode": "observed", "required_fields": ["text"]},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
            "metadata": {"name": "Append literal text"},
        }

        result = execute_tool(
            "set_pipeline",
            args,
            state,
            catalog,
            data_dir=str(tmp_path),
            session_engine=engine,
            session_id=session_id,
            **_verbatim_blob_context(engine, session_id, "hello"),
        )

        assert result.success is True
        assert _default_source(result.updated_state) is not None
        source_options = _default_source(result.updated_state).options
        assert source_options["column"] == "text"
        assert source_options["blob_ref"] == result.data["inline_blob"]["blob_id"]
        assert "hello" not in str(result.to_dict())

        with engine.connect() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == source_options["blob_ref"])).one()
        assert row.filename == "input.txt"
        assert row.mime_type == "text/plain"
        assert row.source_description == "literal input from the user prompt"
        assert Path(row.storage_path).read_text(encoding="utf-8") == "hello"

    def test_set_pipeline_non_gate_with_condition_skips_validation(self) -> None:
        """set_pipeline only validates conditions on gate nodes.

        Transform nodes with a stray condition field are not expression-validated
        (structural validation in CompositionState.validate() catches this).
        """
        state = _empty_state()
        catalog = _mock_catalog()
        args = _valid_pipeline_args()
        # Add a condition to the transform node — structurally wrong but not expression-validated
        args["nodes"][0]["condition"] = "this is garbage syntax!!!"
        result = execute_tool("set_pipeline", args, state, catalog)
        # Succeeds at tool level; validate() flags structural mismatch
        assert result.success is True


# ---------------------------------------------------------------------------
# Failed mutation version contract test
# ---------------------------------------------------------------------------


class TestFailedMutationVersionStable:
    """Failed mutations must not advance the version counter."""

    def test_failed_mutation_preserves_version(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # remove_node with nonexistent ID fails
        result = execute_tool(
            "remove_node",
            {"id": "nonexistent_node"},
            state,
            catalog,
        )
        assert result.success is False
        # Version must not advance on failure
        assert result.updated_state.version == state.version


# ---------------------------------------------------------------------------
# Service-level KeyError handling test
# ---------------------------------------------------------------------------


class TestServiceMissingArgToolArgumentError:
    """Tool raises ToolArgumentError on missing required argument.

    Post Task 4 (set_source manifest promotion), missing required
    arguments are caught at the Tier-3 boundary by
    :class:`SetSourceArgumentsModel`; the handler re-raises
    :class:`pydantic.ValidationError` as :class:`ToolArgumentError` so
    the compose loop's ARG_ERROR routing at ``service.py:2480`` receives
    the right exception class.  A bare ``KeyError`` (the prior shape) is
    a plugin-bug indicator and would be routed to ``PLUGIN_CRASH`` —
    that disposition is wrong for Tier-3 input.
    """

    def test_missing_required_arg_raises_tool_argument_error(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError

        state = _empty_state()
        catalog = _mock_catalog()
        # set_source requires "plugin" — omitting it must raise
        # ToolArgumentError (no longer KeyError) per Task 4.
        with pytest.raises(ToolArgumentError):
            execute_tool("set_source", {}, state, catalog)


# ---------------------------------------------------------------------------
# clear_source tool tests
# ---------------------------------------------------------------------------


class TestClearSource:
    def test_clear_source_removes_source(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # First set a source
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert _default_source(r1.updated_state) is not None
        # Now clear it
        r2 = execute_tool("clear_source", {}, r1.updated_state, catalog)
        assert r2.success is True
        assert _default_source(r2.updated_state) is None
        assert r2.updated_state.version == 3

    def test_clear_source_no_source_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("clear_source", {}, state, catalog)
        assert result.success is False
        assert "No source" in result.data["error"]


# ---------------------------------------------------------------------------
# explain_validation_error tool tests
# ---------------------------------------------------------------------------


class TestExplainValidationError:
    def test_explains_no_source(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": "No source configured."},
            state,
            catalog,
        )
        assert result.success is True
        assert "source" in result.data["explanation"].lower()
        assert "set_source" in result.data["suggested_fix"]

    def test_explains_unknown_node_reference(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": "Edge 'e1' references unknown node 'foo' as from_node."},
            state,
            catalog,
        )
        assert result.success is True
        assert "from_node" in result.data["suggested_fix"]

    def test_explains_duplicate_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": "Duplicate node ID: 'transform_1'."},
            state,
            catalog,
        )
        assert result.success is True
        assert "unique" in result.data["explanation"].lower()

    def test_unknown_error_returns_generic(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": "Some completely unknown error."},
            state,
            catalog,
        )
        assert result.success is True
        assert "not in the known pattern" in result.data["explanation"]

    def test_explains_path_violation(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": "Path violation (S2): '/etc/passwd' is outside the allowed directories."},
            state,
            catalog,
        )
        assert result.success is True
        assert "allowed directories" in result.data["explanation"]

    def test_explains_unreachable_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": "Node 't1' input 'foo' is not reachable from any edge or the source on_success."},
            state,
            catalog,
        )
        assert result.success is True
        assert "on_success" in result.data["suggested_fix"].lower()

    def test_explains_schema_contract_violation(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": ("Schema contract violation: 'source' -> 'add_world'. Consumer requires ['text']; producer guarantees [].")},
            state,
            catalog,
        )
        assert result.success is True
        assert "upstream" in result.data["explanation"].lower()
        assert "preview_pipeline" in result.data["suggested_fix"]
        assert "patch_source_options" in result.data["suggested_fix"]
        assert "patch_node_options" in result.data["suggested_fix"]

    def test_explains_sink_schema_contract_violation(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {
                "error_text": (
                    "Schema contract violation: 't1' -> 'output:main'. "
                    "Sink 'main' requires fields: [text]. "
                    "Producer (value_transform) guarantees: []. "
                    "Missing fields: [text]."
                )
            },
            state,
            catalog,
        )
        assert result.success is True
        assert "sink" in result.data["explanation"].lower()
        assert "preview_pipeline" in result.data["suggested_fix"]
        assert "patch_output_options" in result.data["suggested_fix"]
        assert "patch_source_options" in result.data["suggested_fix"]
        assert "patch_node_options" in result.data["suggested_fix"]

    # ---------------------------------------------------------------------
    # "Expected ..." hint surfacing — observation elspeth-obs-eb4509376c.
    #
    # The validator catches schema-spec mistakes and produces strings
    # like ``"... Expected single-key dict like {'field_name': 'type'} ..."``.
    # The handler used to throw away that hint entirely. These tests lock
    # in that the hint is echoed verbatim into ``suggested_fix``.
    # ---------------------------------------------------------------------

    def test_surfaces_expected_hint_when_pattern_matches(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # The "Invalid options for source 'csv'" prefix matches an
        # existing pattern; the inner "Expected ..." span is the actionable
        # hint that the catalogue's static fix doesn't carry.
        error_text = (
            "Invalid options for source 'csv': Field spec at index 0 is a "
            "dict with 2 keys. Expected single-key dict like "
            "{'field_name': 'type'} or a string like 'field_name: type'."
        )
        result = execute_tool(
            "explain_validation_error",
            {"error_text": error_text},
            state,
            catalog,
        )
        assert result.success is True
        # Catalogue fix (from the matched pattern) is preserved.
        assert "patch_source_options" in result.data["suggested_fix"]
        # And the validator hint is appended verbatim.
        assert ("Expected single-key dict like {'field_name': 'type'} or a string like 'field_name: type'.") in result.data["suggested_fix"]

    def test_surfaces_expected_hint_when_no_pattern_matches(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        error_text = (
            "Field spec at index 0 is a dict with 2 keys. Expected "
            "single-key dict like {'field_name': 'type'} or a string "
            "like 'field_name: type'."
        )
        result = execute_tool(
            "explain_validation_error",
            {"error_text": error_text},
            state,
            catalog,
        )
        assert result.success is True
        # Falls through to generic explanation but still surfaces the hint.
        assert "not in the known pattern" in result.data["explanation"]
        assert ("Expected single-key dict like {'field_name': 'type'} or a string like 'field_name: type'.") in result.data["suggested_fix"]

    def test_no_hint_appended_when_expected_substring_absent(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {"error_text": "No source configured."},
            state,
            catalog,
        )
        assert result.success is True
        # Without "Expected " the catalogue fix should be returned
        # unchanged — no synthetic hint, no trailing whitespace artifacts.
        assert result.data["suggested_fix"] == ("Use set_source to configure a source plugin (e.g. csv, json, dataverse).")

    def test_expected_hint_stops_at_sentence_boundary(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        error_text = "Some preamble. Expected an integer. Got a string. Other noise."
        result = execute_tool(
            "explain_validation_error",
            {"error_text": error_text},
            state,
            catalog,
        )
        assert result.success is True
        # The first sentence after "Expected " is what we surface.
        # Trailing "Got a string. Other noise." should not be included.
        assert "Expected an integer." in result.data["suggested_fix"]
        assert "Got a string" not in result.data["suggested_fix"]
        assert "Other noise" not in result.data["suggested_fix"]


# ---------------------------------------------------------------------------
# list_models tool tests
# ---------------------------------------------------------------------------


class TestListModels:
    def test_list_models_returns_provider_summary(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("list_models", {}, state, catalog)
        assert result.success is True
        # Without provider filter, returns provider-grouped summary
        assert "providers" in result.data
        assert "total_models" in result.data

    def test_list_models_with_provider_filter(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # Even if litellm returns an empty list, the filter shouldn't crash
        result = execute_tool(
            "list_models",
            {"provider": "openrouter/"},
            state,
            catalog,
        )
        assert result.success is True
        assert isinstance(result.data["models"], (list, tuple))

    def test_list_models_empty_string_provider_filters_unprefixed(self) -> None:
        """Empty string from provider summary round-trips as a filter for unprefixed models."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "list_models",
            {"provider": ""},
            state,
            catalog,
        )
        assert result.success is True
        # Should enter the filter path, not the summary path
        assert "models" in result.data
        assert "providers" not in result.data

    def test_list_models_summary_uses_empty_string_for_unprefixed(self) -> None:
        """Provider summary uses empty string (not display-only label) for unprefixed models."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("list_models", {}, state, catalog)
        assert result.success is True
        providers = result.data.get("providers", {})
        # Must not contain the non-round-trippable "(no provider)" label
        assert "(no provider)" not in providers

    def test_list_models_openrouter_reads_live_catalog(self) -> None:
        """``provider="openrouter/"`` returns the live-catalog slice, not the bundled litellm slice.

        Closes the loop between composer discovery and validator preflight:
        the validator (``CatalogValueSource`` on ``OpenRouterLLMProviderConfig``)
        rejects model identifiers absent from ``get_catalog_values(MODEL_CATALOG_OPENROUTER)``,
        so the discovery tool MUST advertise the exact same set. A
        regression here re-opens the
        ``ValueSourceValidationError`` class of tutorial-run failure.
        """
        from elspeth.contracts.value_source import get_catalog_values
        from elspeth.plugins.transforms.llm.model_catalog import MODEL_CATALOG_OPENROUTER

        state = _empty_state()
        catalog = _mock_catalog()
        live = get_catalog_values(MODEL_CATALOG_OPENROUTER)
        if not live:
            pytest.skip("OpenRouter catalog empty; cannot verify wiring")
        result = execute_tool(
            "list_models",
            {"provider": "openrouter/", "limit": 1000},
            state,
            catalog,
        )
        assert result.success is True
        returned = set(result.data["models"])
        # Sample-correctness: every advertised model exists in the live catalog.
        assert returned <= live, (
            "list_models advertised openrouter slugs absent from the live catalog "
            f"(strays={returned - live!r}) — composer discovery and validator preflight drifted apart"
        )

    def test_list_models_openrouter_count_reflects_live_catalog(self) -> None:
        """The unfiltered summary advertises the live-catalog count for openrouter.

        Prevents the regression where the summary's ``providers["openrouter"]``
        count came from the stale bundled litellm list while a follow-up
        ``provider="openrouter/"`` filter returned the live (smaller) set.
        """
        from elspeth.contracts.value_source import get_catalog_values
        from elspeth.plugins.transforms.llm.model_catalog import MODEL_CATALOG_OPENROUTER

        state = _empty_state()
        catalog = _mock_catalog()
        live = get_catalog_values(MODEL_CATALOG_OPENROUTER)
        if not live:
            pytest.skip("OpenRouter catalog empty; cannot verify wiring")
        result = execute_tool("list_models", {}, state, catalog)
        assert result.success is True
        providers = result.data["providers"]
        assert providers.get("openrouter") == len(live)


# ---------------------------------------------------------------------------
# get_audit_info tool tests
# ---------------------------------------------------------------------------


class TestGetAuditInfo:
    """``get_audit_info`` returns CONSTANT facts about the Landscape audit.

    The audit backend is operator-managed (``WebSettings.get_landscape_url()``)
    and intentionally not composer-controllable (security fix S1, see
    ``web/composer/yaml_generator.py:179``). This tool exists so the LLM can
    answer "is audit on, can I configure it?" without inventing a backend
    type and without leaking operator-internal config (URL, encryption key)
    into chat. These tests pin down both behaviours.
    """

    def test_returns_enabled_true_and_not_modifiable(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("get_audit_info", {}, state, catalog)
        assert result.success is True
        assert result.data["enabled"] is True
        assert result.data["composer_modifiable"] is False

    def test_summary_fields_present_and_non_empty(self) -> None:
        """The model paraphrases ``summary``; an empty string here would let it invent text."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("get_audit_info", {}, state, catalog)
        assert isinstance(result.data["summary"], str) and result.data["summary"].strip()
        assert isinstance(result.data["audit_export_summary"], str) and result.data["audit_export_summary"].strip()

    def test_payload_never_leaks_operator_internal_fields(self) -> None:
        """Regression guard: never expose URL/DSN/path/encryption-key fields.

        If a future edit adds any of these as a top-level key OR embeds them
        into one of the existing string fields, this test fails. The audit
        URL is operator-internal — surfacing it to the LLM would defeat the
        purpose of S1 (the LLM might echo it back to the user, log it to
        chat history, or use it to construct a sink shape).
        """
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("get_audit_info", {}, state, catalog)
        forbidden_keys = {
            "url",
            "dsn",
            "path",
            "database_url",
            "landscape_url",
            "encryption_key",
            "encryption_key_env",
            "passphrase",
            "backend",
        }
        assert forbidden_keys.isdisjoint(result.data.keys()), (
            f"get_audit_info leaked operator-internal keys into payload: {forbidden_keys & result.data.keys()}"
        )

        # Also string-scan the values: a future change might inline a path
        # into the summary text. Catch obvious DSN / sqlite path patterns.
        joined_text = " ".join(v for v in result.data.values() if isinstance(v, str)).lower()
        for forbidden_substring in ("sqlite:///", "postgresql://", "postgres://", ".db", "/audit", "audit.db"):
            assert forbidden_substring not in joined_text, (
                f"get_audit_info summary text contains forbidden DSN/path substring "
                f"{forbidden_substring!r} — operator-internal config must not be "
                f"echoed into the LLM context (security fix S1)."
            )

    def test_state_is_unchanged(self) -> None:
        """Discovery tool MUST NOT mutate state — updated_state is the same instance."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("get_audit_info", {}, state, catalog)
        assert result.updated_state is state

    def test_is_registered_as_cacheable_discovery(self) -> None:
        """Result is constant per compose call — should be cacheable."""
        from elspeth.web.composer.tools import (
            is_cacheable_discovery_tool,
            is_discovery_tool,
        )

        assert is_discovery_tool("get_audit_info")
        assert is_cacheable_discovery_tool("get_audit_info")

    def test_appears_in_tool_definitions(self) -> None:
        """The tool must be advertised in the LLM-visible function list."""
        from elspeth.web.composer.tools import get_tool_definitions

        names = {defn["name"] for defn in get_tool_definitions()}
        assert "get_audit_info" in names


# ---------------------------------------------------------------------------
# get_plugin_assistance tool tests
# ---------------------------------------------------------------------------


class TestGetPluginAssistance:
    def test_returns_structured_payload_for_known_issue_code(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "web_scrape",
                "issue_code": "web_scrape.content.compact_text",
            },
            state,
            catalog,
        )
        assert result.success is True
        # ToolResult.to_dict deep-thaws ``data`` for LLM consumption; tests
        # exercise the wire shape rather than the frozen in-memory form.
        payload = result.to_dict()["data"]
        assert payload["plugin_type"] == "transform"
        assert payload["plugin_name"] == "web_scrape"
        assert payload["issue_code"] == "web_scrape.content.compact_text"
        assert "summary" in payload
        assert isinstance(payload["summary"], str)
        assert payload["summary"]
        assert isinstance(payload["suggested_fixes"], list)
        assert payload["suggested_fixes"]
        assert isinstance(payload["examples"], list)
        # web_scrape declares two PluginAssistanceExample entries for this code.
        assert len(payload["examples"]) == 2
        for example in payload["examples"]:
            assert isinstance(example["title"], str)
            # before/after are dicts when present (post-thaw).
            assert example["before"] is None or isinstance(example["before"], dict)
            assert example["after"] is None or isinstance(example["after"], dict)
        assert isinstance(payload["composer_hints"], list)

    def test_line_explode_assistance_returns_structured_payload(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "line_explode",
                "issue_code": "line_explode.source_field.line_framed_text",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.data
        assert payload["plugin_name"] == "line_explode"
        assert payload["issue_code"] == "line_explode.source_field.line_framed_text"
        assert payload["summary"]
        assert payload["suggested_fixes"]

    def test_batch_distribution_profile_assistance_points_categorical_counts_to_top_k(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "batch_distribution_profile",
                "issue_code": "batch_distribution_profile.value_field.numeric",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.data
        assert payload["plugin_name"] == "batch_distribution_profile"
        assert payload["issue_code"] == "batch_distribution_profile.value_field.numeric"
        assert "numeric" in payload["summary"]
        assert any("batch_top_k" in fix for fix in payload["suggested_fixes"])

    def test_unknown_issue_code_returns_explicit_no_assistance(self) -> None:
        """Plugin returns None for unknown issue codes -> tool returns success
        with explicit summary=None and empty suggestion list so the agent sees
        that nothing was published rather than a hard failure."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "web_scrape",
                "issue_code": "web_scrape.unrecognized.code",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        assert payload["plugin_name"] == "web_scrape"
        assert payload["issue_code"] == "web_scrape.unrecognized.code"
        assert payload["summary"] is None
        assert payload["suggested_fixes"] == []
        assert payload["examples"] == []
        assert payload["composer_hints"] == []

    def test_unknown_plugin_name_returns_failure(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "no_such_plugin_xyz",
                "issue_code": "anything",
            },
            state,
            catalog,
        )
        assert result.success is False
        assert "no_such_plugin_xyz" in result.data["error"]

    def test_invalid_plugin_type_returns_failure(self) -> None:
        """plugin_type is validated up-front; mistyped value surfaces as failure."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transformer",  # typo
                "plugin_name": "web_scrape",
            },
            state,
            catalog,
        )
        assert result.success is False
        assert "transformer" in result.data["error"]

    def test_dispatches_to_source_family(self) -> None:
        """plugin_type='source' looks up the plugin via get_source_by_name."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "source",
                "plugin_name": "csv",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        assert payload["plugin_type"] == "source"
        assert payload["plugin_name"] == "csv"

    def test_csv_discovery_explains_generated_source_review_handoff(self) -> None:
        """CSV guidance should make source-level invented-source review mechanics explicit."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "source",
                "plugin_name": "csv",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        hints = " ".join(payload["composer_hints"])

        assert "columns tells CSVSource how to parse headerless rows" in hints
        assert "downstream DAG validation still needs a schema guarantee" in hints
        assert "schema.guaranteed_fields" in hints
        assert "CSV source options do not have url_field" in hints
        assert "set url_field on the web_scrape node" in hints
        assert "If you authored CSV rows or chose source values" in hints
        assert "blob-backed source" in hints
        assert "stage invented_source on source.options.interpretation_requirements" in hints
        assert "request_interpretation_review" in hints
        assert "affected_node_id='source'" in hints
        assert "llm_draft equal to the exact CSV text" in hints
        assert "source is not a transform node" in hints
        assert "do not search nodes[] for source" in hints

    def test_dispatches_to_sink_family(self) -> None:
        """plugin_type='sink' looks up the plugin via get_sink_by_name."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "sink",
                "plugin_name": "json",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        assert payload["plugin_type"] == "sink"
        assert payload["plugin_name"] == "json"

    def test_json_sink_discovery_explains_that_sink_names_do_not_clean_rows(self) -> None:
        """JSON sink guidance should point row cleanup back to field_mapper."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "sink",
                "plugin_name": "json",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        hints = " ".join(payload["composer_hints"])

        assert "JSON sink writes the row it receives" in hints
        assert "schema, format, sink name, and output name do not drop fields" in hints
        assert "Use field_mapper before the sink" in hints
        assert "web_scrape results saved without raw page bodies" in hints
        assert "field_mapper(select_only=true)" in hints
        assert "a sink named cleanup is not a cleanup transform" in hints

    def test_omitting_issue_code_returns_discovery_payload(self) -> None:
        """Discovery-time mode: ``issue_code`` may be omitted entirely."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "web_scrape",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        assert payload["issue_code"] is None
        hints = " ".join(payload["composer_hints"])
        assert "transform" in hints
        assert "not a source" in hints
        assert "url_field" in hints
        assert "Unknown source plugin: web_scrape" in hints
        assert "surface prompt-injection shielding as an important recommendation" in hints
        assert "Recommendation is not permission to add a node" in hints
        assert "do not add passthrough, placeholder, no-op, or renamed utility nodes" in hints
        assert "do not substitute azure_content_safety" in hints
        assert "do not insert it automatically" in hints
        assert "route through azure_content_safety first" not in hints

    def test_web_scrape_discovery_explains_pass_through_url_contract(self) -> None:
        """web_scrape should guide downstream nodes to use guaranteed URL fields."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "web_scrape",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        hints = " ".join(payload["composer_hints"])

        assert "passes through upstream row fields" in hints
        assert "fetch_url_final" in hints
        assert "Do not make downstream LLM templates require a URL field" in hints
        assert "unless the upstream source schema or web_scrape schema guarantees that field" in hints
        assert "do not patch web_scrape guaranteed_fields by guess" in hints
        assert "schema is required" in hints
        assert "For raw HTML, set format to raw" in hints
        assert "not html" in hints
        assert "If the user-facing output should exclude raw scraped content" in hints
        assert "route the final path through field_mapper with select_only: true" in hints
        assert "a sink name or output name is not cleanup" in hints
        assert "A validator-valid direct route from web_scrape or an LLM to the sink is still incomplete" in hints
        assert "If scraped public internet content flows into an LLM" in hints
        assert "azure_prompt_shield" in hints
        assert "only when discovery lists it" in hints
        assert "prompt_injection_shield_recommendation" in hints

    def test_field_mapper_discovery_explains_cleanup_whitelist_semantics(self) -> None:
        """field_mapper hints should make utility-cleanup behavior explicit."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "field_mapper",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        assert payload["issue_code"] is None
        hints = " ".join(payload["composer_hints"])

        assert "Use select_only: true" in hints
        assert "whitelist exactly the saved output fields" in hints
        assert "only utility transform that actually removes raw fields" in hints
        assert "source -> web_scrape -> llm -> field_mapper(cleanup) -> sink" in hints
        assert "A JSON sink named cleanup is not a cleanup transform" in hints
        assert "A validator-valid direct route from web_scrape or an LLM to the sink is still incomplete" in hints
        assert "A cleanup stream name is not a cleanup node" in hints
        assert "do not offer to repair it later" in hints
        assert "set the upstream LLM or scraper on_success to the cleanup mapper" in hints
        assert "set the cleanup mapper on_success to the sink" in hints
        assert "If an LLM routes directly to a JSON sink whose name sounds like cleanup" in hints
        assert "the LLM passes through raw scrape fields until this field_mapper whitelists them" in hints
        assert "route the mapper directly to the existing sink" in hints
        assert "do not remove the cleanup mapper or output" in hints
        assert "before raw scraped fields exist cannot satisfy scraped-content cleanup" in hints
        assert "preserve requested enrichment, extraction, scoring, or LLM response fields" in hints
        assert "stage a pipeline_decision interpretation requirement" in hints
        assert "request its review after mutation succeeds" in hints
        assert "naming a sink or output" in hints
        assert "does not clean data" in hints
        assert "If the user already asked to remove, drop, exclude, or avoid saving raw scrape fields" in hints
        assert "that request is the authorization and requirement to add the cleanup field_mapper" in hints
        assert "do not ask whether to add cleanup later" in hints
        assert "use user_term 'drop_raw_html_fields'" in hints
        assert "not permission to omit the cleanup node" in hints

    def test_llm_discovery_recommends_prompt_shield_for_internet_content(self) -> None:
        """LLM plugin hints recommend prompt shield without silently changing topology."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_assistance",
            {
                "plugin_type": "transform",
                "plugin_name": "llm",
            },
            state,
            catalog,
        )
        assert result.success is True
        payload = result.to_dict()["data"]
        assert payload["issue_code"] is None
        hints = " ".join(payload["composer_hints"])
        assert "reviewed for EVERY LLM node" in hints
        assert "surfaced as an advisory (never blocking)" in hints
        assert "azure_prompt_shield" in hints
        assert "only when discovery lists it" in hints
        assert "whenever no authorized shield is upstream (State B/C)" in hints
        assert "recommendation is not permission to add a node" in hints
        assert "do not add passthrough, placeholder, no-op, or renamed utility nodes" in hints
        assert "copying it verbatim" in hints
        assert "llm_prompt_template" in hints
        assert "scoring scale" in hints
        assert "Measurable adjectives are not exempt" in hints
        assert "over 6 ft" in hints
        assert "top quartile" in hints
        assert "Prompt-template review is not enough" in hints
        assert "put both interpretation_requirements in the LLM node options before set_pipeline" in hints
        assert "When repairing or upserting an LLM node, repeat the review preflight" in hints
        assert (
            "carry forward existing pending LLM interpretation requirements and add missing vague_term or prompt shield requirements"
            in hints
        )
        assert "If the prompt asks the model to return a score, rating, rank, class, or pass/fail result" in hints
        assert "that output shape is authored judgement semantics when you chose the scale" in hints
        assert "stable user_term preserving the user's criterion phrase" in hints
        assert "not the whole task phrase" in hints
        assert "use the adjective or noun phrase that names the criterion" in hints
        assert "For how <adjective> phrasing, use the adjective itself as user_term" in hints
        assert "only an llm_prompt_template review is incomplete" in hints
        assert "Do not stop by saying the rubric is part of the reviewed prompt" in hints
        assert "downstream cleanup, sink, mapper, or transform needs the LLM response" in hints
        assert "guarantee the response_field by name" in hints
        assert "also guarantee pass-through fields" in hints
        assert "Single-query LLM output is written to response_field" in hints
        assert "Prompt-requested JSON keys are not separate pipeline fields unless another transform parses them" in hints
        assert "preserve response_field through cleanup" in hints
        assert "preserves upstream row fields while adding response_field" in hints
        assert "does not remove raw scrape fields" in hints
        assert "put a field_mapper cleanup node between the LLM and the sink" in hints
        assert "rate how cool they are" not in hints
        assert "vague_term" in hints
        assert "good, bad" not in hints
        assert "Subjective user terms" not in hints
        assert "do not insert it automatically" in hints
        assert "prompt_injection_shield_recommendation" in hints
        assert "LLM-node reviews stack" in hints
        assert "Interpretation reviews are not transform stages" in hints
        assert "Do not create passthrough, review, recommendation, or placeholder nodes" in hints
        assert "route through azure_content_safety first" not in hints

    def test_llm_post_call_hints_do_not_keyword_scan_subjective_terms(self) -> None:
        """Plugin hints must not decide arbitrary/vague terms by backend word list."""
        from elspeth.plugins.transforms.llm.transform import LLMTransform

        hints = LLMTransform.get_post_call_hints(
            tool_name="upsert_node",
            config_snapshot={
                "prompt_template": "Rate how cool this public web page is on a 1-10 scale.",
                "response_field": "cool_assessment",
            },
        )

        assert hints == ()


# ---------------------------------------------------------------------------
# preview_pipeline tool tests
# ---------------------------------------------------------------------------


class TestPreviewPipeline:
    def test_preview_empty_pipeline(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("preview_pipeline", {}, state, catalog)
        assert result.success is True
        assert result.data["is_valid"] is False
        assert _pipeline_state_default_source(result.data) is None
        assert result.data["node_count"] == 0

    def test_preview_valid_pipeline(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # Build a minimal valid pipeline
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "t1",
                "on_success": "main",
                "options": {"schema": {"mode": "observed"}},
            },
            r1.updated_state,
            catalog,
        )
        r3 = execute_tool(
            "set_output",
            {"sink_name": "main", "plugin": "csv", "options": {"path": "/data/outputs/output.csv", "schema": {"mode": "observed"}}},
            r2.updated_state,
            catalog,
        )
        result = execute_tool("preview_pipeline", {}, r3.updated_state, catalog)
        assert result.success is True
        assert _pipeline_state_default_source(result.data)["plugin"] == "csv"
        assert _pipeline_state_default_source(result.data)["has_schema_config"] is True
        assert result.data["node_count"] == 1
        assert result.data["output_count"] == 1

    def test_preview_pipeline_includes_edge_contracts(self) -> None:
        """preview_pipeline includes raw edge contract evidence from validation."""
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "fixed", "fields": ["text: str"]}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "value_transform",
                "input": "t1",
                "on_success": "main",
                "on_error": "discard",
                "options": {
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text']"}],
                    "schema": {"mode": "observed"},
                },
            },
            r1.updated_state,
            catalog,
        )
        r3 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r2.updated_state,
            catalog,
        )

        result = execute_tool("preview_pipeline", {}, r3.updated_state, catalog)

        assert result.success is True
        assert "edge_contracts" in result.data
        contracts = result.data["edge_contracts"]
        assert len(contracts) >= 1
        source_to_t1 = next(c for c in contracts if c["to"] == "t1")
        assert source_to_t1["from"] == "source"
        assert "text" in source_to_t1["producer_guarantees"]
        assert "text" in source_to_t1["consumer_requires"]
        assert source_to_t1["satisfied"] is True
        assert result.data["is_valid"] is True

    def test_preview_pipeline_suggests_fork_gate_for_duplicate_consumers(self) -> None:
        """Duplicate consumers get a copyable fork-gate repair skeleton."""
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="classified_rows",
                    options={"path": "/data/in.csv", "schema": {"mode": "fixed", "fields": ["text: str"]}},
                    on_validation_failure="quarantine",
                )
            )
            .with_node(
                NodeSpec(
                    id="fraud_filter",
                    node_type="transform",
                    plugin="value_transform",
                    input="classified_rows",
                    on_success="fraud_rows",
                    on_error="discard",
                    options={
                        "operations": [{"target": "fraud_flag", "expression": "row['text']"}],
                        "schema": {"mode": "observed"},
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
            .with_node(
                NodeSpec(
                    id="regular_filter",
                    node_type="transform",
                    plugin="value_transform",
                    input="classified_rows",
                    on_success="regular_rows",
                    on_error="discard",
                    options={
                        "operations": [{"target": "regular_flag", "expression": "row['text']"}],
                        "schema": {"mode": "observed"},
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
            .with_output(OutputSpec(name="fraud_rows", plugin="csv", options={"path": "outputs/fraud.csv"}, on_write_failure="discard"))
            .with_output(OutputSpec(name="regular_rows", plugin="csv", options={"path": "outputs/regular.csv"}, on_write_failure="discard"))
        )

        result = execute_tool("preview_pipeline", {}, state, _mock_catalog())
        payload = result.to_dict()["data"]

        assert result.success is True
        assert any("Duplicate consumer for connection 'classified_rows'" in err["message"] for err in payload["errors"])
        repairs = payload["graph_repair_suggestions"]
        assert len(repairs) == 1
        repair = repairs[0]
        assert repair["code"] == "duplicate_consumer_connection"
        assert repair["connection"] == "classified_rows"
        assert repair["strategy"] == "insert_fork_gate"
        assert repair["tool_sequence"][0]["arguments"]["input"] == "classified_rows_to_fraud_filter"
        assert repair["tool_sequence"][1]["arguments"]["input"] == "classified_rows_to_regular_filter"
        assert repair["tool_sequence"][2] == {
            "tool": "upsert_node",
            "arguments": {
                "id": "fork_classified_rows",
                "node_type": "gate",
                "plugin": None,
                "input": "classified_rows",
                "on_success": None,
                "on_error": None,
                "options": {},
                "condition": "True",
                "routes": {},
                "fork_to": ["classified_rows_to_fraud_filter", "classified_rows_to_regular_filter"],
                "branches": None,
                "policy": None,
                "merge": None,
                "trigger": None,
                "output_mode": None,
                "expected_output_count": None,
            },
        }
        assert repair["tool_sequence"][-1] == {"tool": "preview_pipeline", "arguments": {}}

        fixed_state = state
        for step in repair["tool_sequence"][:-1]:
            step_result = execute_tool(step["tool"], step["arguments"], fixed_state, _mock_catalog())
            fixed_state = step_result.updated_state
        fixed_preview = execute_tool("preview_pipeline", {}, fixed_state, _mock_catalog()).to_dict()["data"]

        assert not any("Duplicate consumer for connection 'classified_rows'" in err["message"] for err in fixed_preview["errors"])

    def test_preview_source_with_schema_config_field_name(self) -> None:
        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv",
                on_success="t1",
                options={"path": "/data/in.csv", "schema_config": {"mode": "observed"}},
                on_validation_failure="quarantine",
            )
        )
        catalog = _mock_catalog()

        result = execute_tool("preview_pipeline", {}, state, catalog)

        assert result.success is True
        assert _pipeline_state_default_source(result.data)["plugin"] == "csv"
        assert _pipeline_state_default_source(result.data)["has_schema_config"] is True

    def test_preview_pipeline_surfaces_runtime_preflight_failure(self) -> None:
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="main",
                    options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                    on_validation_failure="discard",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        catalog = _mock_catalog()
        runtime_preflight = MagicMock(
            return_value=ValidationResult(
                is_valid=False,
                checks=[
                    ValidationCheck(
                        name="settings_load",
                        passed=False,
                        detail="Forbidden name: 'end_of_source'",
                        affected_nodes=(),
                        outcome_code=None,
                    )
                ],
                errors=[
                    ValidationError(
                        component_id="agg1",
                        component_type="aggregation",
                        message="Forbidden name: 'end_of_source'",
                        suggestion="Omit trigger for end-of-source-only aggregation.",
                        error_code=None,
                    )
                ],
                readiness=ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[]),
            )
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            catalog,
            data_dir="/data",
            runtime_preflight=runtime_preflight,
        )

        assert result.success is True
        assert result.runtime_preflight is not None
        assert result.data["authoring_validation"]["is_valid"] is True
        assert result.data["runtime_preflight"]["is_valid"] is False
        assert result.data["is_valid"] is False
        assert result.data["runtime_preflight"]["errors"][0]["message"] == "Forbidden name: 'end_of_source'"
        runtime_preflight.assert_called_once_with(state)

    def test_preview_pipeline_without_runtime_preflight_preserves_authoring_validation(self) -> None:
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="main",
                    options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                    on_validation_failure="discard",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            data_dir="/data",
            runtime_preflight=None,
        )

        assert result.success is True
        assert result.runtime_preflight is None
        assert result.data["authoring_validation"]["is_valid"] is True
        assert result.data["runtime_preflight"] is None
        assert result.data["is_valid"] is True


class TestPrevalidatePluginOptions:
    """Direct unit tests for _prevalidate_plugin_options.

    Covers the 6 code paths identified in the architecture review:
    structured rejection (unknown plugin), success, config error with prefix stripping,
    injected_fields merge, MappingProxyType deep-thaw, and ValueError surfacing.
    Also covers the absence-is-evidence contract: missing required fields (like
    path) must produce validation errors, not be papered over by placeholders.
    """

    def test_valid_options_returns_none(self) -> None:
        """Valid config returns None (no pre-validation error)."""
        result = _prevalidate_plugin_options(
            "transform",
            "passthrough",
            {"schema": {"mode": "observed"}},
        )
        assert result is None

    def test_invalid_options_returns_error_string(self) -> None:
        """Missing required field returns a descriptive error string."""
        result = _prevalidate_plugin_options(
            "transform",
            "passthrough",
            {},  # missing required 'schema'
        )
        assert result is not None
        assert result.startswith("Invalid options for transform 'passthrough':")

    def test_unknown_transform_plugin_returns_actionable_error(self) -> None:
        """Unregistered transform name surfaces as an actionable error string."""
        result = _prevalidate_plugin_options(
            "transform",
            "this_plugin_does_not_exist",
            {"some": "options"},
        )
        assert result is not None
        assert "this_plugin_does_not_exist" in result
        assert "transform" in result.lower()
        assert "list_transforms" in result

    def test_unknown_source_plugin_returns_actionable_error(self) -> None:
        """Unregistered source name surfaces as an actionable error string."""
        result = _prevalidate_plugin_options(
            "source",
            "no_such_source_plugin",
            {"path": "/data/blobs/in.csv"},
        )
        assert result is not None
        assert "no_such_source_plugin" in result
        assert "source" in result.lower()
        assert "list_sources" in result

    def test_unknown_sink_plugin_returns_actionable_error(self) -> None:
        """Unregistered sink name surfaces as an actionable error string."""
        result = _prevalidate_plugin_options(
            "sink",
            "no_such_sink_plugin",
            {"path": "/data/outputs/out.csv"},
        )
        assert result is not None
        assert "no_such_sink_plugin" in result
        assert "sink" in result.lower()
        assert "list_sinks" in result

    def test_injected_fields_satisfy_required_options(self) -> None:
        """Injected fields are merged in for validation only — not stored in state."""
        # csv source requires on_validation_failure + path (injected) plus schema (in options)
        result = _prevalidate_plugin_options(
            "source",
            "csv",
            {"schema": {"mode": "observed"}},
            injected_fields={"on_validation_failure": "discard", "path": "/tmp/test.csv"},
        )
        assert result is None

    def test_frozen_mappingproxy_options_are_thawed(self) -> None:
        """MappingProxyType options from CompositionState are deep-thawed before Pydantic sees them."""
        from types import MappingProxyType

        frozen_options = MappingProxyType({"schema": MappingProxyType({"mode": "observed"})})
        result = _prevalidate_plugin_options(
            "transform",
            "passthrough",
            frozen_options,  # type: ignore[arg-type]
        )
        assert result is None

    def test_config_class_prefix_stripped_from_error(self) -> None:
        """'Invalid configuration for XConfig:' prefix is stripped so the LLM sees only the problem."""
        result = _prevalidate_plugin_options(
            "transform",
            "passthrough",
            {},  # missing required 'schema'
        )
        assert result is not None
        # Internal class name should not appear — LLM gets the validation detail only
        assert "Invalid configuration for PassThroughConfig" not in result
        assert result.startswith("Invalid options for transform 'passthrough':")

    def test_llm_unknown_provider_surfaced_not_swallowed(self) -> None:
        """ValueError from get_config_model (unknown LLM provider) becomes an error, not silent None."""
        result = _prevalidate_plugin_options(
            "transform",
            "llm",
            {"provider": "nonexistent_provider", "schema": {"mode": "observed"}},
        )
        assert result is not None
        assert result.startswith("Invalid options for transform 'llm':")
        assert "nonexistent_provider" in result

    def test_llm_valid_provider_missing_required_fields_surfaces_errors(self) -> None:
        """Valid provider with missing required fields reports them — not silent None.

        Verifies Phase 2 of LLM dispatch: after provider resolution succeeds,
        the provider-specific Pydantic model validates required fields. Without
        this test, a regression that returns the base LLMConfig (which lacks
        deployment_name/endpoint) instead of AzureOpenAIConfig would be invisible.
        """
        result = _prevalidate_plugin_options(
            "transform",
            "llm",
            {"provider": "azure", "schema": {"mode": "observed"}},
        )
        assert result is not None
        assert result.startswith("Invalid options for transform 'llm':")
        # Azure-specific required fields must be reported
        assert "deployment_name" in result
        assert "endpoint" in result
        assert "api_key" in result
        assert "template" in result

    def test_llm_openrouter_missing_required_fields_surfaces_errors(self) -> None:
        """OpenRouter with missing required fields reports provider-specific missing fields."""
        result = _prevalidate_plugin_options(
            "transform",
            "llm",
            {"provider": "openrouter", "schema": {"mode": "observed"}},
        )
        assert result is not None
        assert result.startswith("Invalid options for transform 'llm':")
        # OpenRouter-specific required fields — model is required (no deployment_name fallback)
        assert "model" in result
        assert "api_key" in result
        assert "template" in result

    def test_llm_openrouter_invalid_model_surfaces_list_models_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Composer prevalidation must reject unknown OpenRouter models with a repair hint.

        Catalog membership is now enforced via the value-source walker
        (check_config_value_sources), so patch the catalog at the walker's lookup
        site rather than the (removed) construction-time validator's.
        """
        monkeypatch.setattr(
            "elspeth.engine.orchestrator.preflight.get_catalog_values",
            lambda catalog_id: frozenset({"openai/gpt-4o"}),
        )

        result = _prevalidate_plugin_options(
            "transform",
            "llm",
            {
                "provider": "openrouter",
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "prompt_template": "Analyze: {{ row.text }}",
                "schema": {"mode": "observed"},
                "required_input_fields": [],
            },
        )
        assert result is not None
        assert result.startswith("Invalid options for transform 'llm':")
        assert "list_models" in result
        assert "anthropic/claude-3-opus" in result

    def test_unreachable_plugin_type_raises_assertion(self) -> None:
        """Passing an invalid plugin_type triggers the unreachable-branch assertion (not silent bypass)."""
        with pytest.raises(AssertionError, match="unexpected plugin_type"):
            _prevalidate_plugin_options(
                "unknown_kind",  # type: ignore[arg-type]
                "csv",
                {},
            )

    def test_absent_source_path_returns_error_not_none(self) -> None:
        """Absence of path is evidence of a missing required field — not a gap to fill with a placeholder.

        Pre-validates csv source options without path. The function must return a descriptive
        error, not None. This guards against regression where callers inject a fake placeholder
        path to suppress the error (violating the data manifesto's 'absence is evidence' rule).
        """
        result = _prevalidate_plugin_options(
            "source",
            "csv",
            {"schema": {"mode": "observed"}},
            injected_fields={"on_validation_failure": "quarantine"},
            # path deliberately absent — caller did not provide it
        )
        assert result is not None
        assert "path" in result

    def test_absent_sink_path_returns_error_not_none(self) -> None:
        """Absence of path for a sink plugin is a validation error, not a placeholder opportunity.

        Pre-validates csv sink options without path. The function must return a descriptive
        error, not None. Regression guard for the symmetric case on sinks.
        """
        result = _prevalidate_plugin_options(
            "sink",
            "csv",
            {},
            # path deliberately absent, no injected_fields
        )
        assert result is not None
        assert "path" in result

    def test_null_source_no_config_model_returns_none(self) -> None:
        """NullSource is registered as None in the source registry — no config validation needed.

        Exercises the ``config_cls is None`` branch in
        ``_prevalidate_plugin_options``. This is distinct from
        UnknownPluginTypeError: 'null' is a known, valid plugin that explicitly has no config
        class (it is a resume-only source with no options).
        """
        result = _prevalidate_plugin_options(
            "source",
            "null",
            {},
        )
        assert result is None

    def test_batch_stats_valid_options(self) -> None:
        """Aggregation plugin (batch_stats) with valid options passes pre-validation.

        batch_stats is an aggregation plugin dispatched as plugin_type="transform".
        This exercises the aggregation-as-transform path in _prevalidate_plugin_options.
        """
        result = _prevalidate_plugin_options(
            "transform",
            "batch_stats",
            {
                "schema": {"mode": "observed"},
                "value_field": "amount",
            },
        )
        assert result is None

    def test_batch_stats_missing_value_field(self) -> None:
        """Aggregation plugin with missing required field returns error."""
        result = _prevalidate_plugin_options(
            "transform",
            "batch_stats",
            {
                "schema": {"mode": "observed"},
                # value_field deliberately absent
            },
        )
        assert result is not None
        assert result.startswith("Invalid options for transform 'batch_stats':")
        assert "value_field" in result

    def test_batch_stats_empty_value_field_rejected(self) -> None:
        """batch_stats rejects empty string value_field via field_validator."""
        result = _prevalidate_plugin_options(
            "transform",
            "batch_stats",
            {
                "schema": {"mode": "observed"},
                "value_field": "",
            },
        )
        assert result is not None
        assert "value_field" in result

    def test_upsert_node_aggregation_type_validates_as_transform(self) -> None:
        """upsert_node with node_type='aggregation' routes through _prevalidate_transform.

        Regression guard: the upsert_node guard checks
        ``node_type in ("transform", "aggregation")``. If someone narrowed this
        to ``node_type == "transform"``, aggregation nodes would bypass
        pre-validation. This test exercises the aggregation path end-to-end.
        """
        state = _empty_state()
        catalog = _mock_catalog()
        # Add batch_stats to the mock catalog's transform list
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="batch_stats",
                description="Batch statistics aggregation",
                plugin_type="transform",
                config_fields=[],
            ),
        ]
        # Missing value_field should be caught by pre-validation
        result = execute_tool(
            "upsert_node",
            {
                "id": "agg1",
                "node_type": "aggregation",
                "plugin": "batch_stats",
                "input": "source",
                "on_success": "out",
                "options": {"schema": {"mode": "observed"}},
                # value_field deliberately absent
            },
            state,
            catalog,
        )
        assert result.success is False

    def test_upsert_node_aggregation_valid_options_succeeds(self) -> None:
        """upsert_node with node_type='aggregation' and valid options succeeds."""
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="batch_stats",
                description="Batch statistics aggregation",
                plugin_type="transform",
                config_fields=[],
            ),
        ]
        result = execute_tool(
            "upsert_node",
            {
                "id": "agg1",
                "node_type": "aggregation",
                "plugin": "batch_stats",
                "input": "source",
                "on_success": "out",
                "options": {
                    "schema": {"mode": "observed"},
                    "value_field": "amount",
                },
            },
            state,
            catalog,
        )
        assert result.success is True
        node = result.updated_state.nodes[0]
        assert node.id == "agg1"
        assert node.node_type == "aggregation"
        assert node.plugin == "batch_stats"

    def test_upsert_node_batch_stats_group_by_keeps_expected_count_open(self) -> None:
        """Grouped rollups may emit one row per group, so cardinality stays unset."""
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="batch_stats",
                description="Batch statistics aggregation",
                plugin_type="transform",
                config_fields=[],
            ),
        ]
        result = execute_tool(
            "upsert_node",
            {
                "id": "agg1",
                "node_type": "aggregation",
                "plugin": "batch_stats",
                "input": "source",
                "on_success": "out",
                "options": {
                    "schema": {"mode": "observed"},
                    "value_field": "amount",
                    "group_by": "customer_tier",
                },
            },
            state,
            catalog,
        )
        assert result.success is True
        node = result.updated_state.nodes[0]
        assert node.expected_output_count is None

    def test_upsert_node_aggregation_rejects_required_input_fields(self) -> None:
        """ADR-013 declared input fields are not valid for batch-aware aggregation nodes."""
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="batch_stats",
                description="Batch statistics aggregation",
                plugin_type="transform",
                config_fields=[],
            ),
        ]

        result = execute_tool(
            "upsert_node",
            {
                "id": "agg1",
                "node_type": "aggregation",
                "plugin": "batch_stats",
                "input": "source",
                "on_success": "out",
                "options": {
                    "schema": {"mode": "observed"},
                    "value_field": "amount",
                    "required_input_fields": ["amount"],
                },
            },
            state,
            catalog,
        )

        assert result.success is False
        assert result.data is not None
        messages = result.data["error"]
        assert "required_input_fields" in messages
        assert "batch-aware" in messages

    def test_upsert_node_rejects_batch_replicate_as_plain_transform(self) -> None:
        """Batch-only fan-out plugins must use the aggregation path, not row-mode transforms."""
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.list_transforms.return_value = [
            *catalog.list_transforms.return_value,
            PluginSummary(
                name="batch_replicate",
                description="Batch replicate aggregation",
                plugin_type="transform",
                config_fields=[],
            ),
        ]

        result = execute_tool(
            "upsert_node",
            {
                "id": "replicate",
                "node_type": "transform",
                "plugin": "batch_replicate",
                "input": "source",
                "on_success": "out",
                "on_error": "discard",
                "options": {
                    "schema": {"mode": "observed"},
                    "replications": [
                        {"source_field": "region", "output_field": "region_copy"},
                    ],
                },
            },
            state,
            catalog,
        )

        assert result.success is False
        assert result.data is not None
        messages = result.data["error"]
        assert "batch_replicate" in messages
        assert "aggregation" in messages
        assert "output_mode: transform" in messages

    def test_secret_ref_field_passes_prevalidation(self) -> None:
        """Options with secret_ref markers pass prevalidation.

        A secret-ref'd field is provisioned (the user called wire_secret_ref),
        just deferred to execution time. Prevalidation must not reject it.
        """
        result = _prevalidate_plugin_options(
            "transform",
            "azure_content_safety",
            {
                "api_key": {"secret_ref": "AZURE_API_KEY"},
                "endpoint": "https://test.cognitiveservices.azure.com",
                "schema": {"mode": "observed"},
                "fields": "text",
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
            },
        )
        assert result is None

    def test_secret_ref_field_non_secret_errors_still_reported(self) -> None:
        """Secret-ref'd fields are forgiven but other errors are still reported.

        api_key has a secret_ref (valid), but fields and thresholds are missing
        (real errors). The error message must mention the missing fields but
        NOT api_key.
        """
        result = _prevalidate_plugin_options(
            "transform",
            "azure_content_safety",
            {
                "api_key": {"secret_ref": "AZURE_API_KEY"},
                "endpoint": "https://test.cognitiveservices.azure.com",
                "schema": {"mode": "observed"},
                # fields and thresholds deliberately absent
            },
        )
        assert result is not None
        assert "fields" in result
        assert "thresholds" in result
        assert "api_key" not in result

    def test_secret_ref_in_source_passes_prevalidation(self) -> None:
        """Source options with a secret_ref marker pass prevalidation."""
        result = _prevalidate_plugin_options(
            "transform",
            "llm",
            {
                "provider": "openrouter",
                "model": "openai/gpt-4o",
                "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                "prompt_template": "Classify: {{text}}",
                "schema": {"mode": "observed"},
            },
        )
        assert result is None

    def test_malformed_secret_ref_marker_still_reports_field_error(self) -> None:
        """Only syntactically valid secret_ref markers are stripped.

        A non-string secret_ref value is malformed and must remain in the
        options payload so the plugin config model reports the field error
        during composer-time validation.
        """
        result = _prevalidate_plugin_options(
            "transform",
            "azure_content_safety",
            {
                "api_key": {"secret_ref": 123},
                "endpoint": "https://test.cognitiveservices.azure.com",
                "schema": {"mode": "observed"},
                "fields": "text",
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
            },
        )
        assert result is not None
        assert "api_key" in result


# ---------------------------------------------------------------------------
# create_blob Tier-3 type guard (elspeth-7a26880c65, Task 2)
# ---------------------------------------------------------------------------


class TestCreateBlobTypeGuard:
    """The Tier-3 content-type guard must raise ToolArgumentError, not TypeError.

    Mocked service-level tests (test_wrong_type_tool_arg_returns_error in
    test_service.py) patch execute_tool at the seam and cannot prove the
    real handler raises the right class. This test drives the handler
    end-to-end through execute_tool() dispatch.

    Post Task 13 (Wave 2 — ``create_blob`` manifest promotion): the
    Tier-3 type guard now lives in :class:`CreateBlobArgumentsModel`
    via Pydantic's strict ``content: str`` validation.  The handler
    catches :class:`pydantic.ValidationError` and re-raises as
    :class:`ToolArgumentError` (pattern at ``tools.py:2320-2327`` /
    Task 4); the LLM-facing message names the argument-bundle and the
    Pydantic model, not the offending value (rev-2 BLOCKER_A leak
    discipline).  Memory:
    ``feedback_locked_in_buggy_expectations`` — the previous
    ``"'content' must be a string, got int"`` assertion pinned the
    legacy ``_prepare_blob_create`` message; the new boundary fires
    earlier with a leak-safe shell.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

    def test_non_string_content_raises_tool_argument_error(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from elspeth.web.composer.protocol import ToolArgumentError

        catalog = _mock_catalog()
        state = _empty_state()

        # Post Task 13 the LLM-facing message names the argument-bundle and
        # the Pydantic model; the raw offending value is not echoed (the
        # full Pydantic detail survives on __cause__ for auditors).
        with pytest.raises(
            ToolArgumentError,
            match=r"'create_blob arguments' must be object conforming to CreateBlobArgumentsModel, got ValidationError",
        ) as exc_info:
            execute_tool(
                "create_blob",
                {
                    "filename": "notes.txt",
                    "mime_type": "text/plain",
                    "content": 42,  # wrong type — LLM sent int where str required
                },
                state,
                catalog,
                data_dir=str(self.data_dir),
                session_engine=self.engine,
                session_id=self.session_id,
            )
        # __cause__ chain MUST preserve the structured Pydantic detail
        # (rev-2 BLOCKER_A leak discipline: full detail audit-side,
        # leak-safe shell LLM-side).
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)


# ---------------------------------------------------------------------------
# update_blob Tier-3 type guard (elspeth-7a26880c65, Task 3)
# ---------------------------------------------------------------------------


class TestUpdateBlobTypeGuard:
    """Parallels TestCreateBlobTypeGuard for _execute_update_blob.

    The fixture is deliberately copy-pasted from TestCreateBlobTypeGuard
    rather than factored into a shared helper: the two guards are
    independent raise sites and one should be moveable without the other.

    Post Task 13 (Wave 2 — ``update_blob`` manifest promotion): the
    Tier-3 type guard now lives in :class:`UpdateBlobArgumentsModel`
    via Pydantic's strict ``content: str`` validation.  Identical
    discipline to :class:`TestCreateBlobTypeGuard` — the same locked-in
    legacy assertion (``"'content' must be a string, got int"``) is
    updated to the new ToolArgumentError shape; the file-mutation
    critical section is never entered on a pure argument-validation
    failure (handler docstring pins this precedence).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

    def test_non_string_content_raises_tool_argument_error(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from elspeth.web.composer.protocol import ToolArgumentError

        catalog = _mock_catalog()
        state = _empty_state()

        # Seed a real blob so the handler reaches the content guard before
        # the "blob not found" check.  Use the create path end-to-end so
        # the row is persisted by the same code path production uses.
        create_result = execute_tool(
            "create_blob",
            {"filename": "a.txt", "mime_type": "text/plain", "content": "initial"},
            state,
            catalog,
            data_dir=str(self.data_dir),
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "initial"),
        )
        blob_id = create_result.data["blob_id"]
        state = create_result.updated_state

        # Post Task 13 the LLM-facing message names the argument-bundle and
        # the Pydantic model; the raw offending value is not echoed (the
        # full Pydantic detail survives on __cause__ for auditors).
        with pytest.raises(
            ToolArgumentError,
            match=r"'update_blob arguments' must be object conforming to UpdateBlobArgumentsModel, got ValidationError",
        ) as exc_info:
            execute_tool(
                "update_blob",
                {"blob_id": blob_id, "content": 42},
                state,
                catalog,
                data_dir=str(self.data_dir),
                session_engine=self.engine,
                session_id=self.session_id,
            )
        # __cause__ chain MUST preserve the structured Pydantic detail.
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)


# ---------------------------------------------------------------------------
# set_source_from_blob Tier-3 type guard
# ---------------------------------------------------------------------------


class TestSetSourceFromBlobTypeGuard:
    """Malformed `options` must stay a retryable tool-argument failure.

    Post Task 13 (Wave 2 — ``set_source_from_blob`` manifest promotion):
    the Tier-3 type guard now lives in
    :class:`SetSourceFromBlobArgumentsModel` via Pydantic's strict
    ``options: dict[str, Any]`` validation.  The handler catches
    :class:`pydantic.ValidationError` and re-raises as
    :class:`ToolArgumentError` (pattern at ``tools.py:2320-2327`` /
    Task 4); the LLM-facing message names the argument-bundle and the
    Pydantic model, not the offending value (rev-2 BLOCKER_A leak
    discipline).  Memory:
    ``feedback_locked_in_buggy_expectations`` — the previous
    ``"'options' must be an object, got str"`` assertion pinned the
    legacy in-handler isinstance guard; the new boundary fires earlier
    with a leak-safe shell.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

    def test_non_object_options_raises_tool_argument_error(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from elspeth.web.composer.protocol import ToolArgumentError

        catalog = _mock_catalog()
        state = _empty_state()

        create_result = execute_tool(
            "create_blob",
            {"filename": "seed.txt", "mime_type": "text/plain", "content": "hello"},
            state,
            catalog,
            data_dir=str(self.data_dir),
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "hello"),
        )
        blob_id = create_result.data["blob_id"]
        state = create_result.updated_state

        # Post Task 13 the LLM-facing message names the argument-bundle and
        # the Pydantic model; the raw offending value (the string
        # "column=text") and its type ("got str") are not echoed.
        with pytest.raises(
            ToolArgumentError,
            match=r"'set_source_from_blob arguments' must be object conforming to SetSourceFromBlobArgumentsModel, got ValidationError",
        ) as exc_info:
            execute_tool(
                "set_source_from_blob",
                {
                    "blob_id": blob_id,
                    "on_success": "out",
                    "options": "column=text",
                },
                state,
                catalog,
                data_dir=str(self.data_dir),
                session_engine=self.engine,
                session_id=self.session_id,
            )
        # __cause__ chain MUST preserve the structured Pydantic detail.
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)


# ---------------------------------------------------------------------------
# get_blob_content — Tier-1 guards (bug_002: composer tool bypassed the
# lifecycle / integrity / decode guards enforced by
# BlobServiceImpl.read_blob_content).  Any path that returns blob bytes
# to the LLM must refuse partial/failed blobs, detect corruption or
# tampering via hash verification, and not crash the tool dispatcher on
# non-UTF-8 bytes that the MIME allowlist happens to admit.
# ---------------------------------------------------------------------------


class TestGetBlobContentGuards:
    """``get_blob_content`` must mirror BlobServiceImpl.read_blob_content guards.

    The composer tool returns blob bytes to an LLM composing a pipeline.
    Without these guards the LLM can:

    * observe a partially-written blob (status=pending) and treat it as
      authoritative;
    * observe a blob whose on-disk bytes have drifted from the stored
      content_hash (corruption, tampering, or a write-path bug) without
      the Tier-1 BlobIntegrityError firing;
    * crash the tool dispatcher with an unhandled UnicodeDecodeError on
      non-UTF-8 bytes that the MIME allowlist admits (``text/csv`` is
      frequently latin-1 in the wild).

    The canonical read path — ``BlobServiceImpl.read_blob_content`` —
    is async and engine-bound, so the fix mirrors its three guards
    inline.  These tests pin the guard semantics so future drift is
    caught at CI time.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        now = datetime.now(UTC)

        # Real content with a real SHA-256 so hash verification can
        # succeed on the happy path and be perturbed deterministically
        # on the mismatch path.
        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_data.csv"
        self.content_bytes = b"col_a,col_b\n1,2\n3,4\n"
        self.content_hash_hex = _content_hash(self.content_bytes)
        self.storage_path.write_bytes(self.content_bytes)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.content_bytes),
                    content_hash=self.content_hash_hex,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def _set_status(self, status: str) -> None:
        from elspeth.web.sessions.models import blobs_table

        with self.engine.begin() as conn:
            conn.execute(blobs_table.update().where(blobs_table.c.id == self.blob_id).values(status=status))

    def test_ready_blob_with_matching_hash_returns_content(self) -> None:
        """Happy path — status=ready, hash matches, bytes are UTF-8."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_blob_content",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert result.data["content"] == self.content_bytes.decode("utf-8")

    def test_pending_blob_refused(self) -> None:
        """Status guard — pending blobs may be partial writes."""
        self._set_status("pending")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_blob_content",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "pending" in result.data["error"].lower() or "not readable" in result.data["error"].lower()

    def test_error_blob_refused(self) -> None:
        """Status guard — error blobs belong to failed runs and are not trustworthy."""
        # The blobs CHECK constraint disallows reading ready→error without a hash,
        # but error is a valid status value; flip it directly.
        self._set_status("error")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_blob_content",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "error" in result.data["error"].lower() or "not readable" in result.data["error"].lower()

    def test_hash_mismatch_raises_blob_integrity_error(self) -> None:
        """Integrity guard — corruption/tampering must ESCALATE, not return failure.

        Tier-1 policy: a mismatch between on-disk bytes and stored
        content_hash is a Tier-1 anomaly (our hash, our file — a
        mismatch means corruption, tampering, or a write-path bug).
        Downgrading to a tool-failure result tells the LLM "retry",
        masking a live audit-integrity event.
        """
        from elspeth.web.blobs.protocol import BlobIntegrityError

        # Mutate the on-disk bytes without touching the DB — simulates
        # filesystem corruption / tampering.
        self.storage_path.write_bytes(b"col_a,col_b\n9,9\n9,9\n")

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(BlobIntegrityError) as exc_info:
            execute_tool(
                "get_blob_content",
                {"blob_id": self.blob_id},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
            )
        assert exc_info.value.blob_id == self.blob_id

    def test_null_content_hash_on_ready_blob_raises_audit_integrity_error(self) -> None:
        """A ready blob with NULL content_hash is a DB-integrity anomaly.

        The blobs table has CHECK constraints forbidding this state;
        reaching it means the invariant was breached out-of-band.
        Must escalate (Tier-1), not return a tool-failure.
        """
        from sqlalchemy import text

        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.web.sessions.models import blobs_table

        # Bypass the CHECK constraint by dropping then re-inserting via
        # raw SQL — the test exercises the defensive read guard, not
        # the write-side invariant.  Use PRAGMA to disable the
        # constraint temporarily.

        with self.engine.begin() as conn:
            conn.execute(text("PRAGMA ignore_check_constraints = 1"))
            conn.execute(blobs_table.update().where(blobs_table.c.id == self.blob_id).values(content_hash=None))
            conn.execute(text("PRAGMA ignore_check_constraints = 0"))

        state = _empty_state()
        catalog = _mock_catalog()
        with pytest.raises(AuditIntegrityError, match="NULL content_hash"):
            execute_tool(
                "get_blob_content",
                {"blob_id": self.blob_id},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
            )

    def test_non_utf8_bytes_return_failure_not_crash(self) -> None:
        """Decode safety — UnicodeDecodeError must not escape the tool.

        The MIME allowlist admits ``text/csv``, ``text/plain`` etc.
        without constraining encoding.  A latin-1 CSV (common in the
        wild) raises UnicodeDecodeError on ``read_text(encoding='utf-8')``;
        without a decode guard this crashes the tool dispatcher with
        an unhandled exception.  The correct behaviour is a structured
        failure so the compose loop can surface a helpful message.
        """
        from elspeth.web.blobs.service import content_hash as _content_hash

        # Bytes that are valid latin-1 but invalid UTF-8 (0xFE is an
        # invalid leading byte in UTF-8).
        non_utf8_bytes = b"na\xefve,col_b\n1,2\n"
        self.storage_path.write_bytes(non_utf8_bytes)

        # Update the DB hash so integrity check passes — the test
        # targets the decode step, not the integrity step.
        from elspeth.web.sessions.models import blobs_table

        with self.engine.begin() as conn:
            conn.execute(
                blobs_table.update()
                .where(blobs_table.c.id == self.blob_id)
                .values(
                    size_bytes=len(non_utf8_bytes),
                    content_hash=_content_hash(non_utf8_bytes),
                )
            )

        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "get_blob_content",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "utf-8" in result.data["error"].lower()


# ---------------------------------------------------------------------------
# update_blob — active-run guard (bug_004: composer could mutate blob bytes
# while an ExecutionService run was actively consuming them).  Mirrors the
# delete_blob two-check pattern: blob_run_links lookup + composition_states
# source scan for the pre-link window.
# ---------------------------------------------------------------------------


class TestUpdateBlobActiveRunGuard:
    """update_blob must refuse to mutate blobs referenced by active runs.

    Two corruption modes without the guard:

    * Path-based sources: the pipeline reads the new bytes but records
      them under the old content_hash — silent Tier-1 audit corruption.
    * blob_ref sources: a mid-run BlobIntegrityError fires as a
      false-positive tamper event because the recomputed hash no
      longer matches the stored hash.

    Both modes are closed by refusing the update while any active
    (pending/running) run in the blob's session references the blob.
    Mirrors the pattern in _execute_delete_blob so the two mutating
    tools enforce the same invariant.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        self.run_id = str(uuid4())
        now = datetime.now(UTC)

        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_data.csv"
        self.original_content = b"a,b\n1,2"
        self.storage_path.write_bytes(self.original_content)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.original_content),
                    content_hash=_STUB_SHA256,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def _insert_run_and_link(self, status: str) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from elspeth.web.sessions.models import (
            blob_run_links_table,
            composition_states_table,
            runs_table,
        )

        now = datetime.now(UTC)
        state_id = str(uuid4())
        with self.engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=self.session_id,
                    version=1,
                    source=None,
                    nodes=None,
                    edges=None,
                    outputs=None,
                    metadata_=None,
                    is_valid=False,
                    validation_errors=None,
                    # Plan §2294: composer-tools test fixture; provenance
                    # required for setup row supporting subsequent runs/
                    # blob_run_links FKs.
                    provenance="session_seed",
                    created_at=now,
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=self.run_id,
                    session_id=self.session_id,
                    state_id=state_id,
                    status=status,
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                )
            )
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=self.blob_id,
                    run_id=self.run_id,
                    direction="input",
                )
            )

    def _insert_run_without_link(self, status: str, *, source: dict[str, Any] | None = None) -> None:
        """Simulate the pre-link window (run exists, blob_run_links row not yet inserted)."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        if source is None:
            source = {
                "plugin": "csv",
                "on_success": "output",
                "on_validation_failure": "quarantine",
                "options": {"blob_ref": self.blob_id, "path": str(self.storage_path)},
            }
        else:
            source = {
                "on_success": "output",
                "on_validation_failure": "quarantine",
                **source,
            }

        now = datetime.now(UTC)
        state_id = str(uuid4())
        with self.engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=self.session_id,
                    version=1,
                    source=source,
                    nodes=[],
                    edges=[],
                    outputs=[],
                    metadata_={"name": "Test", "description": ""},
                    is_valid=False,
                    validation_errors=None,
                    # Plan §2294: composer-tools test fixture; provenance
                    # required for setup row supporting subsequent runs/
                    # blob_run_links FKs.
                    provenance="session_seed",
                    created_at=now,
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=self.run_id,
                    session_id=self.session_id,
                    state_id=state_id,
                    status=status,
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                )
            )

    def test_update_succeeds_when_no_runs_exist(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "new,content\n9,9"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new,content\n9,9"),
        )
        assert result.success is True
        assert self.storage_path.read_bytes() == b"new,content\n9,9"

    def test_update_without_authoring_context_fails_closed(self) -> None:
        from elspeth.contracts.errors import AuditIntegrityError

        state = _empty_state()
        catalog = _mock_catalog()

        with pytest.raises(AuditIntegrityError, match="missing: user_message_id"):
            execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "new,content\n9,9"},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
            )

        assert self.storage_path.read_bytes() == self.original_content
        with self.engine.begin() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == self.blob_id)).one()
        assert row.content_hash == _STUB_SHA256

    def test_update_rejected_when_blob_is_current_source_blob_ref(self) -> None:
        """A blob-backed source locks the blob content hash stamped in source_authoring."""
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="output",
                options={
                    "blob_ref": self.blob_id,
                    "path": str(self.storage_path),
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        catalog = _mock_catalog()

        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "new,content\n9,9"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new,content\n9,9"),
        )

        assert result.success is False
        assert "source" in result.data["error"].lower()
        assert self.storage_path.read_bytes() == self.original_content
        with self.engine.begin() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == self.blob_id)).one()
        assert row.content_hash == _STUB_SHA256

    def test_unbound_update_recomputes_composer_provenance(self) -> None:
        """Unbound blob updates that author new bytes refresh blob provenance."""
        from elspeth.web.blobs.service import content_hash

        state = _empty_state()
        catalog = _mock_catalog()
        user_message_content = "Generate a replacement CSV for the current scratch blob."
        user_message_id = _insert_user_message(self.engine, self.session_id, user_message_content)
        new_content = "name,score\nada,42\n"

        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": new_content},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            user_message_id=user_message_id,
            user_message_content=user_message_content,
            composer_model_identifier="openai/gpt-5-mini",
            composer_model_version="gpt-5-mini-2026-05-01",
            composer_provider="openai",
            composer_skill_hash="sha256:composer-skill",
            tool_arguments_hash="sha256:update-arguments",
        )

        assert result.success is True
        assert self.storage_path.read_bytes() == new_content.encode("utf-8")
        with self.engine.begin() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == self.blob_id)).one()
        assert row.content_hash == content_hash(new_content.encode("utf-8"))
        assert row.creation_modality == CreationModality.LLM_GENERATED.value
        assert row.created_from_message_id == user_message_id
        assert row.creating_model_identifier == "openai/gpt-5-mini"
        assert row.creating_model_version == "gpt-5-mini-2026-05-01"
        assert row.creating_provider == "openai"
        assert row.creating_composer_skill_hash == "sha256:composer-skill"
        assert row.creating_arguments_hash == "sha256:update-arguments"

    def test_update_rejected_when_pending_run_linked(self) -> None:
        self._insert_run_and_link("pending")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "new"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.read_bytes() == self.original_content, "File must not change when the active-run guard blocks the update"

    def test_update_rejected_when_running_run_linked(self) -> None:
        self._insert_run_and_link("running")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "new"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.read_bytes() == self.original_content

    def test_update_succeeds_when_completed_run_linked(self) -> None:
        """Completed runs have released the blob — update must proceed."""
        self._insert_run_and_link("completed")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "post,run\n1,1"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "post,run\n1,1"),
        )
        assert result.success is True
        assert self.storage_path.read_bytes() == b"post,run\n1,1"

    def test_update_rejected_pre_link_window_blob_ref_source(self) -> None:
        """Pre-link window: run exists, blob_run_links not yet inserted, source uses blob_ref."""
        self._insert_run_without_link("pending")
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "new"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.read_bytes() == self.original_content

    def test_update_rejected_pre_link_window_path_source(self) -> None:
        """Pre-link window: run exists with source.path matching storage_path (no blob_ref)."""
        self._insert_run_without_link(
            "running",
            source={"plugin": "csv", "options": {"path": str(self.storage_path)}},
        )
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "new"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "new"),
        )
        assert result.success is False
        assert "active run" in result.data["error"].lower()
        assert self.storage_path.read_bytes() == self.original_content

    def test_update_succeeds_when_active_run_references_different_source(self) -> None:
        """Unrelated active run (different source) must NOT block update — scoped guard."""
        self._insert_run_without_link(
            "pending",
            source={"plugin": "csv", "options": {"path": "/data/external/unrelated.csv"}},
        )
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "should,proceed\n1,1"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "should,proceed\n1,1"),
        )
        assert result.success is True
        assert self.storage_path.read_bytes() == b"should,proceed\n1,1"


# ---------------------------------------------------------------------------
# update_blob — atomic write-order (bug_004: the file write happened BEFORE
# the DB transaction began, creating a window in which a pipeline reader
# would see new bytes against the stale DB hash even with a correct
# active-run guard).  The fix writes to a sibling tempfile and swaps in
# the new content with os.replace only after the guard + quota + UPDATE
# have all succeeded.
# ---------------------------------------------------------------------------


class TestUpdateBlobAtomicWrite:
    """update_blob must not modify storage_path until DB guards have passed.

    Before the fix, _execute_update_blob called ``write_bytes`` before
    ``session_engine.begin()`` — so any subsequent guard failure (active
    run, quota) forced a rollback-write, and any concurrent reader saw
    new bytes against the stale DB hash in the intervening window.

    The fix serialises the file swap to AFTER guard + quota + UPDATE,
    via ``os.replace(tmp, storage_path)`` inside the transaction.  These
    tests pin that ordering by asserting the storage file is unchanged
    on every guard-rejection path and by exercising a simulated DB
    failure to confirm no orphaned tempfiles remain.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        self.storage_dir = tmp_path / "blobs" / self.session_id
        self.storage_dir.mkdir(parents=True)
        self.storage_path = self.storage_dir / f"{self.blob_id}_data.csv"
        self.original_content = b"ORIGINAL-BYTES"
        self.storage_path.write_bytes(self.original_content)

        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="data.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.original_content),
                    content_hash=_STUB_SHA256,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def test_guard_rejection_leaves_storage_untouched_and_no_tempfile(self) -> None:
        """When active-run guard fires, storage_path bytes are unchanged and no tempfile leaks."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from elspeth.web.sessions.models import (
            blob_run_links_table,
            composition_states_table,
            runs_table,
        )

        # Insert a pending run linked to our blob to force the guard.
        now = datetime.now(UTC)
        run_id = str(uuid4())
        state_id = str(uuid4())
        with self.engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=self.session_id,
                    version=1,
                    source=None,
                    nodes=None,
                    edges=None,
                    outputs=None,
                    metadata_=None,
                    is_valid=False,
                    validation_errors=None,
                    # Plan §2294: composer-tools test fixture; provenance
                    # required for setup row supporting subsequent runs/
                    # blob_run_links FKs.
                    provenance="session_seed",
                    created_at=now,
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=self.session_id,
                    state_id=state_id,
                    status="pending",
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                )
            )
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=self.blob_id,
                    run_id=run_id,
                    direction="input",
                )
            )

        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "update_blob",
            {"blob_id": self.blob_id, "content": "would,corrupt,mid-run\n"},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "would,corrupt,mid-run\n"),
        )
        assert result.success is False
        assert self.storage_path.read_bytes() == self.original_content

        # No sibling tempfiles must remain (stale tempfile accumulation
        # would exhaust inodes and leak uncommitted content).
        leftovers = [p for p in self.storage_dir.iterdir() if p != self.storage_path]
        assert leftovers == [], f"Tempfiles leaked after guard rejection: {leftovers}"

    def test_db_failure_leaves_storage_untouched_and_no_tempfile(self) -> None:
        """Simulated DB failure: storage unchanged, no tempfiles remain.

        After the fix, the file is not written to storage_path until
        ``os.replace`` runs inside the transaction — so a DB failure
        that happens before ``os.replace`` leaves the original bytes
        intact by construction (no rollback-write needed).  The
        tempfile cleanup runs unconditionally in a finally block.
        """
        from unittest.mock import patch

        state = _empty_state()
        catalog = _mock_catalog()
        provenance_context = _verbatim_blob_context(self.engine, self.session_id, "new")

        # Force a DB failure by making begin() raise.  This fires
        # BEFORE any UPDATE / os.replace, so no file mutation can
        # have occurred.
        with (
            patch.object(
                self.engine,
                "begin",
                side_effect=RuntimeError("simulated DB failure"),
            ),
            pytest.raises(RuntimeError, match="simulated DB failure"),
        ):
            execute_tool(
                "update_blob",
                {"blob_id": self.blob_id, "content": "new"},
                state,
                catalog,
                session_engine=self.engine,
                session_id=self.session_id,
                **provenance_context,
            )

        assert self.storage_path.read_bytes() == self.original_content
        leftovers = [p for p in self.storage_dir.iterdir() if p != self.storage_path]
        assert leftovers == [], f"Tempfiles leaked after DB failure: {leftovers}"


# ---------------------------------------------------------------------------
# inspect_source mirrors get_blob_content's lifecycle/integrity/decode
# guards, but returns SourceInspectionFacts as a structured dict rather than
# raw bytes — so the LLM can reason about headers and types without seeing
# row content.
# ---------------------------------------------------------------------------


class TestInspectSourceTool:
    """``inspect_source`` returns bounded structural facts about a blob."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.blob_id = str(uuid4())
        now = datetime.now(UTC)

        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_orders.csv"
        self.content_bytes = b"order_id,customer,price\nO-1,Alice,49.95\nO-2,Bob,150.00\n"
        self.content_hash_hex = _content_hash(self.content_bytes)
        self.storage_path.write_bytes(self.content_bytes)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=self.session_id,
                    filename="orders.csv",
                    mime_type="text/csv",
                    size_bytes=len(self.content_bytes),
                    content_hash=self.content_hash_hex,
                    storage_path=str(self.storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def _set_status(self, status: str) -> None:
        from elspeth.web.sessions.models import blobs_table

        with self.engine.begin() as conn:
            conn.execute(blobs_table.update().where(blobs_table.c.id == self.blob_id).values(status=status))

    def test_returns_csv_facts_for_ready_blob(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "inspect_source",
            {"blob_id": self.blob_id},
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        data = result.data
        assert data["source_kind"] == "csv"
        # Data is deep-frozen by ToolResult.__post_init__: lists become tuples,
        # dicts become MappingProxyType. Compare against the frozen forms.
        assert tuple(data["observed_headers"]) == ("order_id", "customer", "price")
        assert dict(data["inferred_types"]) == {
            "order_id": "str",
            "customer": "str",
            "price": "float",
        }
        assert data["sample_row_count"] == 2
        assert data["redacted_identity"]["filename"] == "orders.csv"
        assert data["redacted_identity"]["mime_type"] == "text/csv"
        # Identity must NOT include storage_path
        assert "storage_path" not in data["redacted_identity"]

    def test_returns_url_candidates_when_present(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.models import blobs_table

        # Add a second blob with URL content
        url_blob_id = str(uuid4())
        url_path = self.storage_path.parent / f"{url_blob_id}_urls.txt"
        url_bytes = b"https://example.com/api\n"
        url_path.write_bytes(url_bytes)
        with self.engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=url_blob_id,
                    session_id=self.session_id,
                    filename="urls.txt",
                    mime_type="text/plain",
                    size_bytes=len(url_bytes),
                    content_hash=_content_hash(url_bytes),
                    storage_path=str(url_path),
                    created_at=datetime.now(UTC),
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

        result = execute_tool(
            "inspect_source",
            {"blob_id": url_blob_id},
            _empty_state(),
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert result.data["source_kind"] == "text"
        # url_candidates are redacted to scheme + host (+ port): userinfo and path
        # are dropped because they can carry credentials / reset tokens / PII.
        assert tuple(result.data["url_candidates"]) == ("https://example.com",)
        assert any("web_scrape" in w for w in result.data["warnings"])

    def test_pending_blob_refused(self) -> None:
        self._set_status("pending")
        result = execute_tool(
            "inspect_source",
            {"blob_id": self.blob_id},
            _empty_state(),
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        # Failure messages live in result.data["error"] (set by _failure_result).
        assert "pending" in result.data["error"].lower()

    def test_missing_blob_returns_failure(self) -> None:
        from uuid import uuid4

        result = execute_tool(
            "inspect_source",
            {"blob_id": str(uuid4())},
            _empty_state(),
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is False
        assert "not found" in result.data["error"].lower()

    def test_without_session_context_returns_failure(self) -> None:
        result = execute_tool(
            "inspect_source",
            {"blob_id": self.blob_id},
            _empty_state(),
            _mock_catalog(),
        )
        assert result.success is False

    def test_hash_mismatch_raises_blob_integrity_error(self) -> None:
        """Tier-1 invariant — corrupted blob must escalate, not return facts."""
        from elspeth.web.blobs.protocol import BlobIntegrityError

        # Tamper with the on-disk bytes so SHA-256 mismatches the stored hash.
        self.storage_path.write_bytes(b"tampered,content\n9,9\n")
        with pytest.raises(BlobIntegrityError):
            execute_tool(
                "inspect_source",
                {"blob_id": self.blob_id},
                _empty_state(),
                _mock_catalog(),
                session_engine=self.engine,
                session_id=self.session_id,
            )

    def test_non_uuid_blob_id_rejected_before_lookup(self) -> None:
        """LLM placeholder identifiers must fail at the blob-id boundary."""
        result = execute_tool(
            "inspect_source",
            {"blob_id": "__missing__"},
            _empty_state(),
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )

        assert result.success is False
        assert "not a valid UUID" in result.data["error"]
        assert "upload" in result.data["error"]
        assert "list_blobs" in result.data["error"]
        assert "not found" not in result.data["error"].lower()


# ---------------------------------------------------------------------------
# preview_pipeline proof step. proof_diagnostics surfaces blocking issues that
# depend on observed input shape: fixed CSV schema omitting observed columns,
# text source containing a URL with no web_scrape downstream, missing or
# unreadable blob storage. is_valid is forced to False when any proof
# diagnostic is blocking, even if authoring/runtime checks pass.
# ---------------------------------------------------------------------------


class TestPreviewProofStep:
    """preview_pipeline must surface input-shape diagnostics when session context is supplied."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.csv_blob_id = str(uuid4())
        self.url_blob_id = str(uuid4())
        now = datetime.now(UTC)

        # CSV blob with three observed columns
        storage_dir = tmp_path / "blobs" / self.session_id
        storage_dir.mkdir(parents=True)
        self.csv_storage_path = storage_dir / f"{self.csv_blob_id}_orders.csv"
        csv_content = b"order_id,customer,price\nO-1,Alice,49.95\nO-2,Bob,150.00\n"
        self.csv_storage_path.write_bytes(csv_content)

        # Text blob with a single URL
        self.url_storage_path = storage_dir / f"{self.url_blob_id}_url.txt"
        url_content = b"https://example.com/data.json\n"
        self.url_storage_path.write_bytes(url_content)

        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.csv_blob_id,
                    session_id=self.session_id,
                    filename="orders.csv",
                    mime_type="text/csv",
                    size_bytes=len(csv_content),
                    content_hash=_content_hash(csv_content),
                    storage_path=str(self.csv_storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )
            conn.execute(
                blobs_table.insert().values(
                    id=self.url_blob_id,
                    session_id=self.session_id,
                    filename="url.txt",
                    mime_type="text/plain",
                    size_bytes=len(url_content),
                    content_hash=_content_hash(url_content),
                    storage_path=str(self.url_storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

    def _state_with_csv_source(
        self,
        *,
        schema_mode: str = "fixed",
        fields: tuple[object, ...] = (),
        on_validation_failure: str = "discard",
    ):
        """Build a state with a CSV blob source via the composer tool API."""
        schema: dict[str, object] = {"mode": schema_mode}
        if fields:
            schema["fields"] = list(fields)

        state = _empty_state()
        catalog = _mock_catalog()
        # Wire source via set_source_from_blob — this is the canonical way to
        # produce a state with source.options.blob_ref set.
        result = execute_tool(
            "set_source_from_blob",
            {
                "blob_id": self.csv_blob_id,
                "on_success": "rows",
                "on_validation_failure": on_validation_failure,
                "options": {"schema": schema},
            },
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success, result.data
        state = result.updated_state

        result = execute_tool(
            "set_output",
            {
                "sink_name": "out",
                "plugin": "json",
                "options": {
                    "path": "outputs/out.json",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success, result.data
        return result.updated_state

    def _state_with_text_url_source(self, *, with_web_scrape: bool):
        """Build a state with a text URL blob source, optionally with web_scrape."""
        state = _empty_state()
        catalog = _mock_catalog()

        result = execute_tool(
            "set_source_from_blob",
            {
                "blob_id": self.url_blob_id,
                "on_success": "url_rows" if with_web_scrape else "content",
                "on_validation_failure": "discard",
                "options": {
                    "column": "url",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                },
            },
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success, result.data
        state = result.updated_state

        if with_web_scrape:
            result = execute_tool(
                "upsert_node",
                {
                    "id": "fetch",
                    "node_type": "transform",
                    "plugin": "web_scrape",
                    "input": "url_rows",
                    "on_success": "content",
                    "on_error": "discard",
                    "options": {
                        "url_field": "url",
                        "schema": {"mode": "fixed", "fields": ["url: str"]},
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                        "format": "text",
                        "text_separator": "\n",
                        "http": {
                            "abuse_contact": "test@example.com",
                            "scraping_reason": "test",
                            "allowed_hosts": "public_only",
                        },
                    },
                },
                state,
                catalog,
            )
            assert result.success, result.data
            state = result.updated_state

        result = execute_tool(
            "set_output",
            {
                "sink_name": "out",
                "plugin": "json",
                "options": {
                    "path": "outputs/out.json",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success, result.data
        return result.updated_state

    # -- Empty / no-op cases ------------------------------------------------

    def test_proof_empty_when_no_blob_source(self) -> None:
        """Path-based source has no blob to inspect — diagnostics empty."""
        # Build a state via set_pipeline using a path-based source (no blob_ref)
        args = _valid_pipeline_args()
        args["source"]["on_validation_failure"] = "discard"
        result = execute_tool("set_pipeline", args, _empty_state(), _mock_catalog())
        assert result.success, result.data

        result = execute_tool(
            "preview_pipeline",
            {},
            result.updated_state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        assert result.data["proof_diagnostics"] == ()

    def test_proof_empty_when_no_session_context(self) -> None:
        """Blob source with no session_engine — proof step degrades to empty."""
        state = self._state_with_csv_source(schema_mode="observed")
        result = execute_tool("preview_pipeline", {}, state, _mock_catalog())
        assert result.success is True
        assert result.data["proof_diagnostics"] == ()

    def test_proof_inspects_named_sources_beyond_compatibility_source(self) -> None:
        """A non-first named source must not bypass proof diagnostics."""
        state = self._state_with_csv_source(schema_mode="observed").with_named_source(
            "url_source",
            SourceSpec(
                plugin="text",
                on_success="content",
                options={
                    "blob_ref": self.url_blob_id,
                    "column": "url",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                },
                on_validation_failure="discard",
            ),
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )

        diagnostics = result.data["proof_diagnostics"]
        matching = [d for d in diagnostics if d["code"] == "text_source_url_without_web_scrape"]
        assert matching
        assert matching[0]["evidence_locator"]["source_name"] == "url_source"

    # -- csv_fixed_schema_omits_observed_columns ----------------------------

    def test_fixed_csv_omits_columns_with_discard_blocks(self) -> None:
        state = self._state_with_csv_source(
            schema_mode="fixed",
            fields=("order_id: str",),
            on_validation_failure="discard",
        )
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        diagnostics = result.data["proof_diagnostics"]
        codes = [d["code"] for d in diagnostics]
        assert "csv_fixed_schema_omits_observed_columns" in codes
        blocking = [d for d in diagnostics if d["severity"] == "blocking"]
        assert blocking, "expected a blocking diagnostic for omitted observed columns"
        # is_valid is forced False by the blocking proof diagnostic.
        assert result.data["is_valid"] is False

    def test_fixed_csv_with_all_columns_does_not_block(self) -> None:
        state = self._state_with_csv_source(
            schema_mode="fixed",
            fields=("order_id: str", "customer: str", "price: float"),
            on_validation_failure="discard",
        )
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "csv_fixed_schema_omits_observed_columns" not in codes

    def test_fixed_csv_with_structured_fields_does_not_crash_or_block(self) -> None:
        state = self._state_with_csv_source(
            schema_mode="fixed",
            fields=(
                {"name": "order_id", "field_type": "str"},
                {"name": "customer", "field_type": "str"},
                {"name": "price", "field_type": "float"},
            ),
            on_validation_failure="discard",
        )
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success is True
        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "csv_fixed_schema_omits_observed_columns" not in codes

    def test_flexible_csv_does_not_block(self) -> None:
        """Flexible mode accepts extra columns by design."""
        state = self._state_with_csv_source(schema_mode="flexible", fields=("order_id: str",))
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "csv_fixed_schema_omits_observed_columns" not in codes

    def test_fixed_csv_with_routed_failures_does_not_block(self) -> None:
        """on_validation_failure routes to a sink → not silent discard, not blocking."""
        state = self._state_with_csv_source(
            schema_mode="fixed",
            fields=("order_id: str",),
            on_validation_failure="quarantine_sink",
        )
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "csv_fixed_schema_omits_observed_columns" not in codes

    # -- text_source_url_without_web_scrape ---------------------------------

    def test_text_url_without_web_scrape_blocks(self) -> None:
        state = self._state_with_text_url_source(with_web_scrape=False)
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        diagnostics = result.data["proof_diagnostics"]
        codes = [d["code"] for d in diagnostics]
        assert "text_source_url_without_web_scrape" in codes
        blocking = [d for d in diagnostics if d["severity"] == "blocking"]
        assert blocking
        assert result.data["is_valid"] is False

    def test_text_url_with_web_scrape_does_not_block(self) -> None:
        state = self._state_with_text_url_source(with_web_scrape=True)
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "text_source_url_without_web_scrape" not in codes

    # -- inspection warnings surfaced as info -------------------------------

    def test_inspection_warnings_surfaced_as_info(self) -> None:
        """The text source's web_scrape warning is mirrored in proof_diagnostics as info."""
        state = self._state_with_text_url_source(with_web_scrape=True)
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        diagnostics = result.data["proof_diagnostics"]
        info = [d for d in diagnostics if d["severity"] == "info"]
        # web_scrape warning from inspection should be mirrored — as info, not blocking
        assert any(d["code"] == "source_inspection_warning" for d in info)

    # -- csv_duplicate_headers (promoted to blocking) -----------------------
    # Duplicate CSV headers cause silent column collapse in csv.DictReader
    # (last-write-wins) and similar libraries, fabricating a single column
    # from multiple source columns. That is a Tier-1 audit-integrity
    # violation and must force the repair loop, not pass through as
    # advisory info.

    def _replace_csv_blob_with_duplicate_headers(self) -> None:
        """Overwrite the seeded CSV blob's bytes + content_hash so it has
        duplicate headers. Must update content_hash to match the new bytes
        or the proof step's BlobIntegrityError check will fire instead.
        """
        from sqlalchemy import update

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.models import blobs_table

        new_bytes = b"order_id,name,name,price\nO-1,Alice,Smith,49.95\nO-2,Bob,Jones,150.00\n"
        self.csv_storage_path.write_bytes(new_bytes)
        with self.engine.begin() as conn:
            conn.execute(
                update(blobs_table)
                .where(blobs_table.c.id == self.csv_blob_id)
                .values(
                    size_bytes=len(new_bytes),
                    content_hash=_content_hash(new_bytes),
                )
            )

    def test_csv_duplicate_headers_blocks(self) -> None:
        """Duplicate CSV headers must surface as a blocking proof diagnostic."""
        self._replace_csv_blob_with_duplicate_headers()
        state = self._state_with_csv_source(schema_mode="observed")
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        diagnostics = result.data["proof_diagnostics"]
        codes = [d["code"] for d in diagnostics]
        assert "csv_duplicate_headers" in codes, diagnostics
        # Severity is blocking, not info.
        dup = next(d for d in diagnostics if d["code"] == "csv_duplicate_headers")
        assert dup["severity"] == "blocking", dup
        # Must carry an actionable suggested_repair string (not None) so the
        # forced-repair loop has a concrete remedy to relay to the LLM.
        assert isinstance(dup["suggested_repair"], str) and dup["suggested_repair"], dup
        # The warning text must reach the LLM verbatim — it names the
        # offending header(s).
        assert "name" in dup["message"], dup
        # is_valid is forced False by the blocking proof diagnostic.
        assert result.data["is_valid"] is False

    def test_csv_without_duplicate_headers_does_not_block(self) -> None:
        """Clean headers must not produce a csv_duplicate_headers diagnostic."""
        state = self._state_with_csv_source(schema_mode="observed")
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "csv_duplicate_headers" not in codes

    def test_observed_csv_numeric_gate_type_mismatch_blocks(self) -> None:
        """Observed CSV leaves values stringy; numeric gate predicates must
        fail preview before runtime hits ExpressionEvaluationError.
        """
        state = self._state_with_csv_source(schema_mode="observed")
        result = execute_tool(
            "upsert_node",
            {
                "id": "price_gate",
                "node_type": "gate",
                "plugin": None,
                "input": "rows",
                "condition": "row['price'] >= 100",
                "routes": {"true": "out", "false": "out"},
                "options": {},
            },
            state,
            _mock_catalog(),
        )
        assert result.success, result.data

        result = execute_tool(
            "preview_pipeline",
            {},
            result.updated_state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )

        diagnostics = result.data["proof_diagnostics"]
        mismatch = [d for d in diagnostics if d["code"] == "gate_expression_type_mismatch_against_source_schema"]
        assert mismatch, diagnostics
        assert mismatch[0]["severity"] == "blocking"
        assert mismatch[0]["evidence_locator"]["node_id"] == "price_gate"
        assert mismatch[0]["evidence_locator"]["field"] == "price"
        assert result.data["is_valid"] is False

    def test_observed_csv_batch_stats_string_value_field_blocks_through_transform(self) -> None:
        """Observed CSV strings must not reach numeric batch_stats at runtime.

        This mirrors the hard-mode p2_t2_edge failure: a string-ish field
        survives an observed-schema value_transform and batch_stats rejects it
        only when the aggregation executes.
        """
        from sqlalchemy import update

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.models import blobs_table

        csv_content = b"respondent_id,community,financial_barrier\nR-1,Community-A,yes\nR-2,Community-B,no\n"
        self.csv_storage_path.write_bytes(csv_content)
        with self.engine.begin() as conn:
            conn.execute(
                update(blobs_table)
                .where(blobs_table.c.id == self.csv_blob_id)
                .values(
                    size_bytes=len(csv_content),
                    content_hash=_content_hash(csv_content),
                )
            )

        state = (
            self._state_with_csv_source(schema_mode="observed")
            .with_node(
                NodeSpec(
                    id="classify",
                    node_type="transform",
                    plugin="value_transform",
                    input="rows",
                    on_success="classified_rows",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "financial_only", "expression": "row['financial_barrier']"}],
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
            .with_node(
                NodeSpec(
                    id="summarize",
                    node_type="aggregation",
                    plugin="batch_stats",
                    input="classified_rows",
                    on_success="out",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "value_field": "financial_barrier",
                        "group_by": "community",
                        "compute_mean": True,
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )

        diagnostics = result.data["proof_diagnostics"]
        mismatch = [d for d in diagnostics if d["code"] == "aggregation_numeric_value_field_type_mismatch_against_source_schema"]
        assert mismatch, diagnostics
        assert mismatch[0]["severity"] == "blocking"
        assert mismatch[0]["evidence_locator"]["node_id"] == "summarize"
        assert mismatch[0]["evidence_locator"]["field"] == "financial_barrier"
        assert mismatch[0]["evidence_locator"]["observed_type"] == "str"
        assert result.data["is_valid"] is False

    def test_observed_named_csv_batch_stats_string_value_field_blocks_through_transform(self) -> None:
        """Named CSV sources must participate in source-field proof walk-back."""
        from sqlalchemy import update

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.models import blobs_table

        csv_content = b"respondent_id,community,financial_barrier\nR-1,Community-A,yes\nR-2,Community-B,no\n"
        self.csv_storage_path.write_bytes(csv_content)
        with self.engine.begin() as conn:
            conn.execute(
                update(blobs_table)
                .where(blobs_table.c.id == self.csv_blob_id)
                .values(
                    size_bytes=len(csv_content),
                    content_hash=_content_hash(csv_content),
                )
            )

        base_state = self._state_with_csv_source(schema_mode="observed")
        named_csv_source = SourceSpec(
            plugin="csv",
            on_success="survey_rows",
            options={
                "blob_ref": self.csv_blob_id,
                "schema": {"mode": "observed"},
            },
            on_validation_failure="discard",
        )
        state = (
            base_state.without_source()
            .with_named_source("survey_csv", named_csv_source)
            .with_node(
                NodeSpec(
                    id="classify",
                    node_type="transform",
                    plugin="value_transform",
                    input="survey_rows",
                    on_success="classified_rows",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "financial_only", "expression": "row['financial_barrier']"}],
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
            .with_node(
                NodeSpec(
                    id="summarize",
                    node_type="aggregation",
                    plugin="batch_stats",
                    input="classified_rows",
                    on_success="out",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "value_field": "financial_barrier",
                        "group_by": "community",
                        "compute_mean": True,
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )

        diagnostics = result.data["proof_diagnostics"]
        mismatch = [d for d in diagnostics if d["code"] == "aggregation_numeric_value_field_type_mismatch_against_source_schema"]
        assert mismatch, diagnostics
        assert mismatch[0]["severity"] == "blocking"
        assert mismatch[0]["evidence_locator"]["node_id"] == "summarize"
        assert mismatch[0]["evidence_locator"]["field"] == "financial_barrier"
        assert mismatch[0]["evidence_locator"]["observed_type"] == "str"
        assert result.data["is_valid"] is False

    def test_observed_csv_numeric_aggregation_does_not_block_after_field_overwrite(self) -> None:
        """The proof step abstains once an upstream transform overwrites the field."""
        from sqlalchemy import update

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.models import blobs_table

        csv_content = b"respondent_id,community,financial_barrier\nR-1,Community-A,yes\nR-2,Community-B,no\n"
        self.csv_storage_path.write_bytes(csv_content)
        with self.engine.begin() as conn:
            conn.execute(
                update(blobs_table)
                .where(blobs_table.c.id == self.csv_blob_id)
                .values(
                    size_bytes=len(csv_content),
                    content_hash=_content_hash(csv_content),
                )
            )

        state = (
            self._state_with_csv_source(schema_mode="observed")
            .with_node(
                NodeSpec(
                    id="coerce_flag",
                    node_type="transform",
                    plugin="value_transform",
                    input="rows",
                    on_success="coerced_rows",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "financial_barrier", "expression": "1"}],
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
            .with_node(
                NodeSpec(
                    id="summarize",
                    node_type="aggregation",
                    plugin="batch_stats",
                    input="coerced_rows",
                    on_success="out",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "value_field": "financial_barrier",
                        "group_by": "community",
                        "compute_mean": True,
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )

        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "aggregation_numeric_value_field_type_mismatch_against_source_schema" not in codes

    def test_csv_duplicate_headers_registered_as_blocking_code(self) -> None:
        """Registry membership ripples — the constructor would crash if the
        emission site used an unregistered code, so this test pins the
        canonical-vocabulary contract independently of emission."""
        from elspeth.web.composer.tools import _BLOCKING_DIAGNOSTIC_CODES

        assert "csv_duplicate_headers" in _BLOCKING_DIAGNOSTIC_CODES
        assert "csv_source_field_resolution_error" in _BLOCKING_DIAGNOSTIC_CODES

    # -- missing/unreadable blob --------------------------------------------

    def test_missing_storage_file_blocks(self) -> None:
        self.csv_storage_path.unlink()
        state = self._state_with_csv_source(schema_mode="observed")
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        codes = [d["code"] for d in result.data["proof_diagnostics"]]
        assert "source_inspection_failed" in codes
        assert result.data["is_valid"] is False

    def test_blocking_proof_overrides_authoring_validation(self) -> None:
        """Authoring may be valid but blocking proof_diagnostics still flips is_valid."""
        state = self._state_with_csv_source(
            schema_mode="fixed",
            fields=("order_id: str",),
            on_validation_failure="discard",
        )
        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        # Stage 1 might be valid, but proof step blocks → is_valid False.
        assert result.data["is_valid"] is False
        # The state-level validation still reflects authoring shape, only
        # the summary-level is_valid is forced. authoring_validation is
        # deep-frozen to MappingProxyType by ToolResult.__post_init__.
        from collections.abc import Mapping as _Mapping

        assert isinstance(result.data["authoring_validation"], _Mapping)

    # -- Tier-3 persisted-option boundaries: malformed source.options ---------
    # ``source.options`` is composer/operator-authored config re-read from
    # persisted session state — Tier-3 origin. A drifted / hand-edited / stale
    # store can carry a malformed value that the helpers re-validate on read,
    # raising ``ValueError``. ``compute_proof_diagnostics`` is called UNWRAPPED
    # from the preview tool and the dispatcher only catches ``ToolArgumentError``,
    # so an unhandled ``ValueError`` would crash the tool. These pin the boundary
    # at the ``compute_proof_diagnostics`` level: each malformed option must yield
    # a blocking diagnostic and NO exception may escape. The tampered options are
    # injected via ``dataclasses.replace`` to model persisted state that drifted
    # after the set_source_from_blob write-time validation that originally
    # produced it (the "validated once is not validated forever" T3 read-back).

    def _state_with_tampered_source_options(self, options: dict[str, object]):
        """Build a valid CSV blob state, then overwrite source.options.

        Models persisted external-origin options that drifted between the
        write-time validation and this read.
        """
        import dataclasses

        state = self._state_with_csv_source(schema_mode="observed")
        source = _default_source(state)
        assert source is not None
        tampered = dataclasses.replace(source, options=options)
        return state.with_source(tampered)

    def test_proof_malformed_schema_block_yields_blocking_diagnostic(self) -> None:
        """A malformed ``schema`` block (get_raw_schema_config raises) blocks,

        does not crash the tool.
        """
        from elspeth.web.composer.tools import compute_proof_diagnostics

        # schema.mode is not a recognised mode → get_raw_schema_config raises
        # ValueError. blob_ref must survive so the proof step inspects the blob.
        state = self._state_with_tampered_source_options({"blob_ref": self.csv_blob_id, "schema": {"mode": "not_a_real_mode"}})
        diagnostics = compute_proof_diagnostics(
            state,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        blocking = [d for d in diagnostics if d["severity"] == "blocking"]
        assert any(d["code"] == "csv_source_field_resolution_error" for d in blocking), diagnostics
        # The schema-block builder's repair text must point at the schema knob,
        # not at header/field_mapping resolution.
        schema_diag = next(d for d in blocking if d["code"] == "csv_source_field_resolution_error")
        assert "schema" in schema_diag["suggested_repair"]
        assert "schema.mode" in schema_diag["suggested_repair"]

    def test_proof_malformed_columns_option_yields_blocking_diagnostic(self) -> None:
        """A malformed ``columns`` option (inspect wrap raises) blocks, no crash."""
        from elspeth.web.composer.tools import compute_proof_diagnostics

        # columns must be a sequence of str; an int member raises ValueError
        # inside _csv_source_columns, surfaced by the inspect-call wrap.
        state = self._state_with_tampered_source_options({"blob_ref": self.csv_blob_id, "columns": ["order_id", 123]})
        diagnostics = compute_proof_diagnostics(
            state,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        blocking = [d for d in diagnostics if d["severity"] == "blocking"]
        assert any(d["code"] == "csv_source_field_resolution_error" for d in blocking), diagnostics

    def test_proof_malformed_delimiter_option_yields_blocking_diagnostic(self) -> None:
        """A malformed ``delimiter`` option (inspect wrap raises) blocks, no crash."""
        from elspeth.web.composer.tools import compute_proof_diagnostics

        # delimiter must be a single character; a multi-char string raises
        # ValueError inside _csv_source_delimiter, surfaced by the inspect wrap.
        state = self._state_with_tampered_source_options({"blob_ref": self.csv_blob_id, "delimiter": "||"})
        diagnostics = compute_proof_diagnostics(
            state,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        blocking = [d for d in diagnostics if d["severity"] == "blocking"]
        assert any(d["code"] == "csv_source_field_resolution_error" for d in blocking), diagnostics

    # -- proof step integrity verification -----------------------------------
    # The proof step reads blob bytes through the same Tier-1 invariants as
    # _execute_get_blob_content and _execute_inspect_source: NULL stored
    # content_hash escalates via AuditIntegrityError; SHA-256 mismatch
    # escalates via BlobIntegrityError. Without these the audit trail would
    # accept LLM repair turns driven by unverified bytes.

    def test_proof_raises_blob_integrity_error_on_hash_mismatch(self) -> None:
        """Tampering with on-disk bytes after upload must raise, not soft-fail.

        The proof step reads the blob's bytes; if SHA-256 of the bytes
        does not match the stored content_hash, that's a Tier-1 anomaly
        (filesystem corruption, tampering, or write-path bug) and must
        ESCALATE — not silently let downstream LLM repair turns act on
        garbage.
        """
        from elspeth.web.blobs.protocol import BlobIntegrityError

        state = self._state_with_csv_source(schema_mode="observed")
        # Tamper with the on-disk bytes after the row was inserted.
        self.csv_storage_path.write_bytes(b"tampered,data\nX,Y\n")
        with pytest.raises(BlobIntegrityError):
            execute_tool(
                "preview_pipeline",
                {},
                state,
                _mock_catalog(),
                session_engine=self.engine,
                session_id=self.session_id,
            )

    def test_proof_raises_audit_integrity_error_on_null_content_hash(self) -> None:
        """A ``ready`` blob with NULL content_hash is a DB-integrity anomaly.

        Enforced at write time by the ``ck_blobs_ready_hash`` CHECK
        constraint. If the proof step ever observes NULL here, the
        constraint was bypassed (or the database is corrupt). Must
        ESCALATE via AuditIntegrityError; cannot soft-degrade to "no
        diagnostics" because that would let an unverified blob drive
        repair turns.
        """
        from sqlalchemy import update

        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.web.sessions.models import blobs_table

        # Bypass the CHECK constraint by suspending it; sqlite respects
        # ``PRAGMA defer_foreign_keys`` but not check toggles inline,
        # so we drop and recreate without the constraint for the test
        # row. Simpler: directly patch via raw SQL with a workaround
        # — but the cleanest test path is to set status='ready' and
        # NULL the hash explicitly via UPDATE; sqlite will reject the
        # CHECK if defined. Use a raw SQL UPDATE that violates the
        # check guard if present, else falls through; the row's
        # presence on read with NULL hash is what we need.
        state = self._state_with_csv_source(schema_mode="observed")
        with self.engine.begin() as conn:
            # Disable the row-level check by toggling pragma; sqlite
            # 3.x respects this for the connection. Then null the hash.
            conn.exec_driver_sql("PRAGMA ignore_check_constraints = ON")
            conn.execute(update(blobs_table).where(blobs_table.c.id == self.csv_blob_id).values(content_hash=None))
            conn.exec_driver_sql("PRAGMA ignore_check_constraints = OFF")

        with pytest.raises(AuditIntegrityError, match="NULL content_hash"):
            execute_tool(
                "preview_pipeline",
                {},
                state,
                _mock_catalog(),
                session_engine=self.engine,
                session_id=self.session_id,
            )


class TestBlockingDiagnosticRegistry:
    """``_blocking_diagnostic`` enforces the canonical-codes invariant.

    The skill markdown that drives the composer LLM cites these codes by
    name; if a contributor adds a new blocker without registering the code
    in ``_BLOCKING_DIAGNOSTIC_CODES``, the LLM's repair vocabulary drifts
    silently. The constructor's runtime assertion turns that drift into a
    crash at the construction site.
    """

    def test_unregistered_code_raises_at_construction(self) -> None:
        from elspeth.web.composer.tools import _blocking_diagnostic

        with pytest.raises(AssertionError, match="not registered in _BLOCKING_DIAGNOSTIC_CODES"):
            _blocking_diagnostic(
                code="this_code_was_never_registered",
                message="msg",
                suggested_repair="repair",
                evidence_locator={},
            )

    def test_registered_codes_construct_successfully(self) -> None:
        from elspeth.web.composer.tools import _BLOCKING_DIAGNOSTIC_CODES, _blocking_diagnostic

        for code in _BLOCKING_DIAGNOSTIC_CODES:
            d = _blocking_diagnostic(
                code=code,
                message="msg",
                suggested_repair="repair",
                evidence_locator={"source": "blob", "blob_id": "abc"},
            )
            # Construction sets severity blocking and preserves the inputs.
            assert d["code"] == code
            assert d["severity"] == "blocking"
            assert d["message"] == "msg"
            assert d["suggested_repair"] == "repair"
            assert d["evidence_locator"] == {"source": "blob", "blob_id": "abc"}


class TestToolContextSecretServiceTyping:
    """Type-contract tests for ``ToolContext.secret_service``.

    Prior to elspeth-d017c958e9 the field was ``Any | None``, which
    silently accepted any object — including the wrong production
    surface (``WebSecretService``, which requires ``auth_provider_type``)
    or a misnamed kwarg. The field now annotates ``WebSecretResolver``
    (L0 protocol from ``elspeth.contracts.secrets``), and production
    wiring passes ``ScopedSecretResolver``.

    These tests pin the structural contract so the dispatch boundary
    stays typed.
    """

    def test_scoped_secret_resolver_satisfies_protocol(self) -> None:
        """Production ``ScopedSecretResolver`` must satisfy the protocol."""
        from elspeth.contracts.secrets import WebSecretResolver
        from elspeth.web.secrets.service import ScopedSecretResolver

        # ScopedSecretResolver only depends on its inner service for
        # delegation; structural-typing checks need only the methods,
        # not a live DB.
        class _ServiceStub:
            def list_refs(self, user_id: str, *, auth_provider_type: str) -> list[Any]:
                return []

            def has_ref(self, user_id: str, name: str, *, auth_provider_type: str) -> bool:
                return False

            def resolve(self, user_id: str, name: str, *, auth_provider_type: str) -> None:
                return None

        resolver = ScopedSecretResolver(_ServiceStub(), auth_provider_type="local")  # type: ignore[arg-type]
        assert isinstance(resolver, WebSecretResolver)

    def test_tool_context_accepts_protocol_implementor(self) -> None:
        """ToolContext.secret_service must accept any WebSecretResolver."""
        from elspeth.web.composer.tools._common import ToolContext

        class _ResolverStub:
            def list_refs(self, user_id: str) -> list[Any]:
                return []

            def has_ref(self, user_id: str, name: str) -> bool:
                return False

            def resolve(self, user_id: str, name: str) -> None:
                return None

        ctx = ToolContext(catalog=MagicMock(spec=CatalogService), secret_service=_ResolverStub())
        # Field is reachable, typed, and forwards the structural surface.
        assert ctx.secret_service is not None
        assert ctx.secret_service.has_ref("u1", "missing") is False
        assert ctx.secret_service.list_refs("u1") == []

    def test_tool_context_secret_service_defaults_to_none(self) -> None:
        """Non-secret-aware callers (legacy direct tests) construct with None."""
        from elspeth.web.composer.tools._common import ToolContext

        ctx = ToolContext(catalog=MagicMock(spec=CatalogService))
        assert ctx.secret_service is None
