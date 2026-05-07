"""Tests for composer MCP server — tool registration and dispatch."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from elspeth.composer_mcp.server import _build_tool_defs, _dispatch_tool
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _invalid_contract_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text']"}],
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
                name="main",
                plugin="csv",
                options={"path": "outputs/out.csv", "schema": {"mode": "observed"}, "collision_policy": "auto_increment"},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _valid_state_with_no_edge_contracts() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="main",
            options={"path": "/data/in.csv", "schema": {"mode": "observed"}},
            on_validation_failure="quarantine",
        ),
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "outputs/out.csv", "schema": {"mode": "observed"}, "collision_policy": "auto_increment"},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _connection_valid_field_mapper_state_without_edges() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="text",
            on_success="mapper_in",
            options={"path": "/data/in.txt", "column": "text", "schema": {"mode": "observed"}},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="map_body",
                node_type="transform",
                plugin="field_mapper",
                input="mapper_in",
                on_success="main",
                on_error="discard",
                options={
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"], "required_fields": ["text"]},
                    "mapping": {"text": "body"},
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
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/out.csv",
                    "schema": {"mode": "observed", "required_fields": ["body"]},
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _mock_catalog() -> CatalogService:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = []
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    return catalog


class TestBuildToolDefs:
    """Tests for _build_tool_defs() tool registration."""

    def test_returns_more_than_20_tools(self) -> None:
        tools = _build_tool_defs()
        assert len(tools) > 20

    def test_tool_count_matches_registry(self) -> None:
        """Tool count must equal composer subset + session tools."""
        from elspeth.composer_mcp.server import _COMPOSER_TOOL_NAMES, _SESSION_TOOL_DEFS

        tools = _build_tool_defs()
        expected = len(_COMPOSER_TOOL_NAMES) + len(_SESSION_TOOL_DEFS)
        assert len(tools) == expected, f"Expected {expected}, got {len(tools)}"

    def test_all_tools_have_name_and_description(self) -> None:
        for tool in _build_tool_defs():
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing 'description'"
            assert tool["name"], "Tool name must be non-empty"
            assert tool["description"], f"Tool {tool['name']} has empty description"

    def test_discovery_tools_present(self) -> None:
        names = {t["name"] for t in _build_tool_defs()}
        for expected in ("list_sources", "list_transforms", "list_sinks", "get_plugin_schema", "get_expression_grammar"):
            assert expected in names, f"Discovery tool '{expected}' missing"

    def test_mutation_tools_present(self) -> None:
        names = {t["name"] for t in _build_tool_defs()}
        for expected in ("set_source", "upsert_node", "upsert_edge", "set_output", "set_pipeline"):
            assert expected in names, f"Mutation tool '{expected}' missing"

    def test_session_tools_present(self) -> None:
        names = {t["name"] for t in _build_tool_defs()}
        for expected in ("new_session", "save_session", "load_session", "list_sessions", "generate_yaml", "delete_session"):
            assert expected in names, f"Session tool '{expected}' missing"

    def test_get_plugin_assistance_tool_registered(self) -> None:
        names = {t["name"] for t in _build_tool_defs()}
        assert "get_plugin_assistance" in names

    def test_blob_and_secret_tools_excluded(self) -> None:
        names = {t["name"] for t in _build_tool_defs()}
        for excluded in (
            "list_blobs",
            "set_source_from_blob",
            "get_blob_metadata",
            "list_secret_refs",
            "validate_secret_ref",
            "wire_secret_ref",
        ):
            assert excluded not in names, f"Blob/secret tool '{excluded}' should be excluded"


class TestDispatchTool:
    """Tests for _dispatch_tool() dispatch logic."""

    @pytest.fixture()
    def scratch_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "scratch"
        d.mkdir()
        return d

    def test_list_sources_returns_success(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "list_sources",
            {},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )
        assert result["success"] is True

    def test_set_source_mutates_state(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "set_source",
            {"plugin": "csv", "on_success": "node_1", "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}}},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )
        assert result["success"] is True
        assert result["state"]["source"]["plugin"] == "csv"

    def test_set_output_requires_explicit_collision_policy(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is False
        assert "collision_policy" in result["error"]
        assert result["state"] == _empty_state().to_dict()

    def test_set_output_accepts_explicit_collision_policy(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {
                    "path": "outputs/out.csv",
                    "schema": {"mode": "observed"},
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is True
        assert result["state"]["outputs"][0]["options"]["collision_policy"] == "auto_increment"

    def test_patch_output_options_cannot_remove_collision_policy(self, scratch_dir: Path) -> None:
        state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={
                        "path": "outputs/out.csv",
                        "schema": {"mode": "observed"},
                        "collision_policy": "auto_increment",
                    },
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = _dispatch_tool(
            "patch_output_options",
            {"sink_name": "main", "patch": {"collision_policy": None}},
            state,
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is False
        assert "collision_policy" in result["error"]
        assert result["state"] == state.to_dict()

    def test_set_pipeline_requires_explicit_output_collision_policy(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "set_pipeline",
            {
                "source": {
                    "plugin": "csv",
                    "on_success": "main",
                    "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                },
                "nodes": [],
                "edges": [],
                "outputs": [
                    {
                        "sink_name": "main",
                        "plugin": "json",
                        "options": {"path": "outputs/out.json", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
            },
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is False
        assert "Output 'main'" in result["error"]
        assert "collision_policy" in result["error"]

    def test_new_session_returns_session_id(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "new_session",
            {},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )
        assert result["success"] is True
        assert "session_id" in result["data"]

    def test_save_and_load_round_trip(self, scratch_dir: Path) -> None:
        # Create a session first
        new_result = _dispatch_tool(
            "new_session",
            {},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )
        session_id = new_result["data"]["session_id"]

        # Modify state via set_source
        modified = _dispatch_tool(
            "set_source",
            {"plugin": "csv", "on_success": "node_1", "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}}},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )
        modified_state = CompositionState.from_dict(modified["state"])

        # Save the modified state
        save_result = _dispatch_tool(
            "save_session",
            {"session_id": session_id},
            modified_state,
            _mock_catalog(),
            scratch_dir,
        )
        assert save_result["success"] is True

        # Load it back
        load_result = _dispatch_tool(
            "load_session",
            {"session_id": session_id},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )
        assert load_result["success"] is True
        assert load_result["state"]["source"]["plugin"] == "csv"

    def test_delete_missing_session_before_scratch_exists_returns_not_found(self, tmp_path: Path) -> None:
        scratch_dir = tmp_path / "scratch"
        session_id = "0" * 12

        result = _dispatch_tool(
            "delete_session",
            {"session_id": session_id},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is False
        assert result["error"] == f"Session not found: {session_id}"
        assert result["state"] == _empty_state().to_dict()

    def test_generate_yaml_returns_string_for_valid_state(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "generate_yaml",
            {},
            _valid_state_with_no_edge_contracts(),
            _mock_catalog(),
            scratch_dir,
        )
        assert result["success"] is True
        assert isinstance(result["data"], str)

    def test_generate_yaml_rejects_state_missing_file_sink_collision_policy(self, scratch_dir: Path) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/data/in.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="json",
                    options={"path": "outputs/out.json", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = _dispatch_tool(
            "generate_yaml",
            {},
            state,
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is False
        assert "collision_policy" in result["error"]

    def test_generate_yaml_rejects_invalid_contract_state(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "generate_yaml",
            {},
            _invalid_contract_state(),
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is False
        assert "invalid" in result["error"].lower()
        assert result["validation"]["is_valid"] is False
        assert len(result["validation"]["errors"]) >= 1
        assert result["validation"]["edge_contracts"] == [
            {
                "from": "source",
                "to": "t1",
                "producer_guarantees": [],
                "consumer_requires": ["text"],
                "missing_fields": ["text"],
                "satisfied": False,
            }
        ]

    def test_generate_yaml_allows_valid_state_with_no_edge_contracts(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "generate_yaml",
            {},
            _valid_state_with_no_edge_contracts(),
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is True
        assert isinstance(result["data"], str)

    def test_generate_yaml_allows_connection_valid_state_without_ui_edges(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "generate_yaml",
            {},
            _connection_valid_field_mapper_state_without_edges(),
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is True
        assert "field_mapper" in result["data"]
        assert "body" in result["data"]

    def test_unknown_tool_returns_failure(self, scratch_dir: Path) -> None:
        result = _dispatch_tool(
            "nonexistent_tool",
            {},
            _empty_state(),
            _mock_catalog(),
            scratch_dir,
        )
        assert result["success"] is False


class TestValidationToDictSemanticContracts:
    """Validation payload must surface semantic_contracts for MCP clients.

    Without this, only the legacy edge_contracts field reaches MCP and
    the new plugin-declared semantic layer is invisible to agent
    consumers — which makes the /validate response asymmetric across
    HTTP and MCP surfaces.
    """

    def test_semantic_contracts_in_payload(self) -> None:
        from elspeth.composer_mcp.server import _validation_to_dict
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ")
        validation = state.validate()
        payload = _validation_to_dict(validation)

        assert "semantic_contracts" in payload
        assert isinstance(payload["semantic_contracts"], list)
        assert len(payload["semantic_contracts"]) == 1
        contract = payload["semantic_contracts"][0]
        assert contract["from_id"] == "scrape"
        assert contract["to_id"] == "explode"
        assert contract["producer_field"] == "content"
        assert contract["consumer_field"] == "content"
        assert contract["outcome"] == "conflict"
        assert contract["consumer_plugin"] == "line_explode"
        assert contract["producer_plugin"] == "web_scrape"
        assert contract["requirement_code"] == "line_explode.source_field.line_framed_text"

    def test_empty_semantic_contracts_emits_empty_list_not_absent(self) -> None:
        """Surface parity: pre-semantic short-circuits emit [], not absent.

        /validate's pre-semantic short-circuit returns serialize the
        Pydantic default of [] rather than omitting the field; MCP must
        match that surface so clients can treat 'absent' as a clear
        signal of an older server version, not as 'maybe satisfied'.
        """
        from elspeth.composer_mcp.server import _validation_to_dict

        state = _empty_state()
        validation = state.validate()
        payload = _validation_to_dict(validation)

        assert "semantic_contracts" in payload
        assert payload["semantic_contracts"] == []


@pytest.mark.asyncio
async def test_mcp_preview_runtime_preflight_joins_shared_session_inflight() -> None:
    from elspeth.composer_mcp.server import _mcp_preview_runtime_preflight
    from elspeth.web.execution.runtime_preflight import RuntimePreflightCoordinator
    from elspeth.web.execution.schemas import ValidationResult

    coordinator = RuntimePreflightCoordinator()
    state = _valid_state_with_no_edge_contracts()
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    expected = ValidationResult(is_valid=True, checks=[], errors=[])

    async def run_preflight(candidate: CompositionState) -> ValidationResult:
        nonlocal calls
        assert candidate is state
        calls += 1
        started.set()
        await release.wait()
        return expected

    first_task = asyncio.create_task(
        _mcp_preview_runtime_preflight(
            state,
            coordinator=coordinator,
            session_scope="session:shared-web-session",
            settings_hash="settings-hash",
            timeout_seconds=1.0,
            run_preflight=run_preflight,
        )
    )
    await started.wait()
    second_task = asyncio.create_task(
        _mcp_preview_runtime_preflight(
            state,
            coordinator=coordinator,
            session_scope="session:shared-web-session",
            settings_hash="settings-hash",
            timeout_seconds=1.0,
            run_preflight=run_preflight,
        )
    )

    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is expected
    assert second is expected
    assert calls == 1
