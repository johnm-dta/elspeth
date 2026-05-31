"""Regression coverage for Landscape MCP startup schema failures."""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
)
from sqlalchemy import create_engine, text

from elspeth.core.landscape.database import LandscapeDB
from elspeth.mcp import server as mcp_server
from elspeth.mcp.server import create_server


def _create_stale_landscape_url(tmp_path: Path) -> str:
    db_path = tmp_path / "stale_audit.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA user_version = 6")
        conn.execute(
            text(
                """
                CREATE TABLE token_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    is_terminal INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL
                )
                """
            )
        )
    engine.dispose()
    return f"sqlite:///{db_path}"


def test_database_descriptor_does_not_swallow_unexpected_url_parser_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only SQLAlchemy parse errors should become the safe invalid-URL marker."""
    import sqlalchemy.engine.url as url_module

    def fail_unexpectedly(database_url: str) -> object:
        raise RuntimeError(f"unexpected parser failure for {database_url}")

    monkeypatch.setattr(url_module, "make_url", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="unexpected parser failure"):
        mcp_server._safe_database_descriptor("sqlite:///audit.db")


@pytest.mark.asyncio
async def test_stale_schema_does_not_abort_mcp_server_creation(tmp_path: Path) -> None:
    """SchemaCompatibilityError must be visible through MCP, not as a broken handshake."""
    server = create_server(_create_stale_landscape_url(tmp_path))

    assert server.instructions is not None
    assert "Landscape database schema is outdated" in server.instructions
    assert "get_mcp_status" in server.instructions
    assert server.create_initialization_options().instructions == server.instructions

    tools_response = await server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    assert isinstance(tools_response.root, ListToolsResult)
    tool_names = {tool.name for tool in tools_response.root.tools}
    assert tool_names == {"get_mcp_status"}


@pytest.mark.asyncio
async def test_ready_server_tool_list_is_json_serializable(tmp_path: Path) -> None:
    """Tool schemas must survive the real MCP JSON serialization boundary."""
    db_path = tmp_path / "audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    db.close()

    server = create_server(f"sqlite:///{db_path}")

    tools_response = await server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    assert isinstance(tools_response.root, ListToolsResult)
    assert len(tools_response.root.tools) > 1

    tools_response.root.model_dump(mode="json", by_alias=True, exclude_none=True)


@pytest.mark.asyncio
async def test_stale_schema_status_tool_returns_mcp_error_result(tmp_path: Path) -> None:
    """The agent must receive the schema problem as an MCP tool error payload."""
    server = create_server(_create_stale_landscape_url(tmp_path))

    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="get_mcp_status", arguments={}),
    )
    response = await server.request_handlers[CallToolRequest](request)

    result = response.root
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.content
    first_content = result.content[0]
    assert isinstance(first_content, TextContent)
    message = first_content.text
    assert "Landscape MCP server cannot open the configured audit database" in message
    assert "Landscape database schema is outdated" in message
    assert "token_outcomes.completed" in message
    assert "docs/operator/migrations/adr-019.md" in message
