# Composer MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the web composer's pipeline-building tools as an MCP server so Claude Code can build and validate pipeline configurations conversationally — using the same tools the web UI's LLM composer uses.

**Architecture:** A stdio MCP server (`elspeth-composer`) that wraps `CatalogServiceImpl` for plugin discovery and `CompositionState` for state mutation/validation. Session state persisted as JSON files in a git-ignored `.scratch/` folder. No Landscape access, no blob tools, no secret tools. Reuses the existing `execute_tool()` dispatcher and `generate_yaml()` from the web composer — no code duplication.

**Tech Stack:** `mcp` (Anthropic MCP SDK), `pluggy` (plugin discovery via existing PluginManager), existing `CompositionState`/`ToolResult`/`generate_yaml` from `web.composer`.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/elspeth/composer_mcp/__init__.py` | Lazy-import facade (mirrors `mcp/__init__.py` pattern) |
| `src/elspeth/composer_mcp/server.py` | MCP server: tool registration, dispatch, CLI entry point |
| `src/elspeth/composer_mcp/session.py` | Session persistence: new/save/load/list from `.scratch/` JSON files |
| `.scratch/.gitkeep` | Git-tracked placeholder for scratch folder |
| `.gitignore` (modify) | Add `.scratch/*` and `!.scratch/.gitkeep` |
| `pyproject.toml` (modify) | Add `elspeth-composer` console script entry point |
| `tests/unit/composer_mcp/test_server.py` | Tool registration and dispatch tests |
| `tests/unit/composer_mcp/test_session.py` | Session persistence round-trip tests |
| `tests/unit/composer_mcp/__init__.py` | Test package marker |

**Layer placement:** `composer_mcp/` is L3 (application layer), same as `mcp/` and `web/`. It imports from L0 (`contracts.freeze`) via `web.composer.state` and from L3 (`web.composer.tools`, `web.composer.yaml_generator`, `web.catalog.service`).

**What we reuse vs. what we write:**

| Reuse from `web.composer` | Write new |
|--------------------------|-----------|
| `CompositionState`, all Spec types | `session.py` — JSON file persistence |
| `execute_tool()` dispatcher + all handlers | `server.py` — MCP protocol machinery |
| `ToolResult` + `ValidationSummary` | `__init__.py` — lazy-import facade |
| `generate_yaml()` | `.scratch/` + `.gitignore` entries |
| `get_expression_grammar()` | `pyproject.toml` script entry |
| `CatalogServiceImpl` + `PluginManager` | Tests |

---

### Task 1: Scratch Folder and Git Ignore

**Files:**
- Create: `.scratch/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create scratch folder with .gitkeep**

```bash
mkdir -p .scratch
touch .scratch/.gitkeep
```

- [ ] **Step 2: Add scratch folder to .gitignore**

Add to `.gitignore`:
```
# Composer MCP scratch folder (session state, test configs)
.scratch/*
!.scratch/.gitkeep
```

- [ ] **Step 3: Verify gitkeep is tracked but contents are ignored**

```bash
git add .scratch/.gitkeep
git status
```

Expected: `.scratch/.gitkeep` is staged, no other `.scratch/` files would be tracked.

- [ ] **Step 4: Commit**

```bash
git add .gitignore .scratch/.gitkeep
git commit -m "chore: add .scratch/ folder for composer MCP sessions"
```

---

### Task 2: Session Persistence

**Files:**
- Create: `src/elspeth/composer_mcp/__init__.py`
- Create: `src/elspeth/composer_mcp/session.py`
- Create: `tests/unit/composer_mcp/__init__.py`
- Create: `tests/unit/composer_mcp/test_session.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/composer_mcp/__init__.py` (empty file).

Create `tests/unit/composer_mcp/test_session.py`:

```python
"""Tests for composer MCP session persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from elspeth.composer_mcp.session import (
    SessionManager,
    SessionNotFoundError,
)
from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec


class TestSessionManager:
    @pytest.fixture()
    def scratch_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "scratch"

    @pytest.fixture()
    def manager(self, scratch_dir: Path) -> SessionManager:
        return SessionManager(scratch_dir)

    def test_new_session_returns_empty_state(self, manager: SessionManager) -> None:
        session_id, state = manager.new_session()
        assert session_id  # non-empty string
        assert state.source is None
        assert state.nodes == ()
        assert state.edges == ()
        assert state.outputs == ()
        assert state.version == 0

    def test_new_session_with_name(self, manager: SessionManager) -> None:
        session_id, state = manager.new_session(name="my-pipeline")
        assert state.metadata.name == "my-pipeline"

    def test_save_and_load_round_trip(self, manager: SessionManager) -> None:
        session_id, state = manager.new_session()
        source = SourceSpec(
            plugin="csv",
            on_success="source_out",
            options={"path": "data/input.csv"},
            on_validation_failure="discard",
        )
        state = state.with_source(source)
        manager.save(session_id, state)

        loaded = manager.load(session_id)
        assert loaded.source is not None
        assert loaded.source.plugin == "csv"
        assert loaded.source.on_success == "source_out"
        assert loaded.version == state.version

    def test_load_nonexistent_raises(self, manager: SessionManager) -> None:
        with pytest.raises(SessionNotFoundError, match="no-such-id"):
            manager.load("no-such-id")

    def test_list_sessions_empty(self, manager: SessionManager) -> None:
        assert manager.list_sessions() == []

    def test_list_sessions_after_save(self, manager: SessionManager) -> None:
        sid1, s1 = manager.new_session(name="first")
        manager.save(sid1, s1)
        sid2, s2 = manager.new_session(name="second")
        manager.save(sid2, s2)

        sessions = manager.list_sessions()
        assert len(sessions) == 2
        names = {s["name"] for s in sessions}
        assert names == {"first", "second"}

    def test_save_creates_scratch_dir(self, tmp_path: Path) -> None:
        scratch = tmp_path / "nonexistent" / "scratch"
        manager = SessionManager(scratch)
        sid, state = manager.new_session()
        manager.save(sid, state)
        assert scratch.exists()

    def test_delete_session(self, manager: SessionManager) -> None:
        sid, state = manager.new_session()
        manager.save(sid, state)
        manager.delete(sid)
        with pytest.raises(SessionNotFoundError):
            manager.load(sid)

    def test_saved_file_is_valid_json(self, manager: SessionManager, scratch_dir: Path) -> None:
        sid, state = manager.new_session(name="json-check")
        manager.save(sid, state)
        files = list(scratch_dir.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["metadata"]["name"] == "json-check"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/composer_mcp/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.composer_mcp'`

- [ ] **Step 3: Write the package init**

Create `src/elspeth/composer_mcp/__init__.py`:

```python
"""MCP server for ELSPETH pipeline composition.

Exposes the same pipeline-building tools as the web composer,
without Landscape access. Uses CatalogService for plugin discovery
and CompositionState for state mutation/validation.
"""


def create_server(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Create a composer MCP server instance. Requires the [mcp] extra."""
    from elspeth.composer_mcp.server import create_server as _create_server

    return _create_server(*args, **kwargs)


def main(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Run the composer MCP server. Requires the [mcp] extra."""
    from elspeth.composer_mcp.server import main as _main

    return _main(*args, **kwargs)


__all__ = ["create_server", "main"]
```

- [ ] **Step 4: Write the session manager**

Create `src/elspeth/composer_mcp/session.py`:

```python
"""Session persistence for the composer MCP server.

Sessions are stored as JSON files in the scratch directory.
Each file contains the serialized CompositionState (via to_dict/from_dict
round-trip) plus session metadata.

Layer: L3 (application). Imports from L3 (web.composer.state).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from elspeth.web.composer.state import CompositionState, PipelineMetadata


class SessionNotFoundError(Exception):
    """Raised when a session ID does not correspond to a saved file."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class SessionManager:
    """Manages CompositionState sessions as JSON files on disk.

    Each session is a single JSON file: ``{scratch_dir}/{session_id}.json``.
    The file contains the full CompositionState serialized via to_dict().
    """

    def __init__(self, scratch_dir: Path) -> None:
        self._dir = scratch_dir

    def new_session(self, *, name: str = "Untitled Pipeline") -> tuple[str, CompositionState]:
        """Create a new empty session. Does not persist until save() is called."""
        session_id = uuid.uuid4().hex[:12]
        state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name=name),
            version=0,
        )
        return session_id, state

    def save(self, session_id: str, state: CompositionState) -> Path:
        """Persist session state to disk. Creates scratch dir if needed."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._session_path(session_id)
        data = state.to_dict()
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return path

    def load(self, session_id: str) -> CompositionState:
        """Load session state from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        data = json.loads(path.read_text())
        return CompositionState.from_dict(data)

    def delete(self, session_id: str) -> None:
        """Delete a saved session."""
        path = self._session_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        path.unlink()

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions with ID, name, and version."""
        if not self._dir.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("*.json")):
            data = json.loads(path.read_text())
            sessions.append({
                "session_id": path.stem,
                "name": data.get("metadata", {}).get("name", "Untitled Pipeline"),
                "version": data.get("version", 0),
            })
        return sessions

    def _session_path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/composer_mcp/test_session.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/composer_mcp/__init__.py src/elspeth/composer_mcp/session.py \
  tests/unit/composer_mcp/__init__.py tests/unit/composer_mcp/test_session.py
git commit -m "feat(composer-mcp): add session persistence for pipeline composition"
```

---

### Task 3: MCP Server — Tool Registration and Dispatch

**Files:**
- Create: `src/elspeth/composer_mcp/server.py`
- Create: `tests/unit/composer_mcp/test_server.py`

This is the core task. The server exposes 21 tools: 8 discovery tools from `_DISCOVERY_TOOLS`, 13 mutation tools from `_MUTATION_TOOLS`, plus 3 session management tools (new_session, save_session, load_session, list_sessions, delete_session, generate_yaml = 6 session/utility tools). Total: 27 tools.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/composer_mcp/test_server.py`:

```python
"""Tests for composer MCP server tool registration and dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from elspeth.composer_mcp.server import (
    _build_tool_defs,
    _dispatch_tool,
)
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.state import CompositionState, PipelineMetadata


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=0,
    )


@pytest.fixture()
def catalog() -> CatalogService:
    mock = MagicMock(spec=CatalogService)
    mock.list_sources.return_value = []
    mock.list_transforms.return_value = []
    mock.list_sinks.return_value = []
    return mock


@pytest.fixture()
def scratch_dir(tmp_path: Path) -> Path:
    return tmp_path / "scratch"


class TestToolDefinitions:
    def test_tool_defs_is_non_empty(self) -> None:
        defs = _build_tool_defs()
        assert len(defs) > 20

    def test_all_tools_have_name_and_description(self) -> None:
        for tool in _build_tool_defs():
            assert tool.name
            assert tool.description

    def test_includes_discovery_tools(self) -> None:
        names = {t.name for t in _build_tool_defs()}
        assert "list_sources" in names
        assert "list_transforms" in names
        assert "list_sinks" in names
        assert "get_plugin_schema" in names
        assert "get_expression_grammar" in names

    def test_includes_mutation_tools(self) -> None:
        names = {t.name for t in _build_tool_defs()}
        assert "set_source" in names
        assert "upsert_node" in names
        assert "upsert_edge" in names
        assert "set_output" in names
        assert "set_pipeline" in names

    def test_includes_session_tools(self) -> None:
        names = {t.name for t in _build_tool_defs()}
        assert "new_session" in names
        assert "save_session" in names
        assert "load_session" in names
        assert "list_sessions" in names
        assert "generate_yaml" in names

    def test_no_blob_or_secret_tools(self) -> None:
        names = {t.name for t in _build_tool_defs()}
        assert "list_blobs" not in names
        assert "set_source_from_blob" not in names
        assert "list_secret_refs" not in names
        assert "wire_secret_ref" not in names


class TestDispatch:
    def test_list_sources_returns_data(
        self, catalog: CatalogService, scratch_dir: Path
    ) -> None:
        result = _dispatch_tool(
            "list_sources", {}, _empty_state(), catalog, scratch_dir
        )
        assert result["success"] is True

    def test_set_source_mutates_state(
        self, catalog: CatalogService, scratch_dir: Path
    ) -> None:
        result = _dispatch_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "source_out",
                "options": {"path": "data/input.csv"},
                "on_validation_failure": "discard",
            },
            _empty_state(),
            catalog,
            scratch_dir,
        )
        assert result["success"] is True
        assert result["state"]["source"]["plugin"] == "csv"

    def test_new_session_returns_id_and_state(
        self, catalog: CatalogService, scratch_dir: Path
    ) -> None:
        result = _dispatch_tool(
            "new_session", {}, _empty_state(), catalog, scratch_dir
        )
        assert result["success"] is True
        assert "session_id" in result["data"]

    def test_save_and_load_round_trip(
        self, catalog: CatalogService, scratch_dir: Path
    ) -> None:
        # Create and save
        new_result = _dispatch_tool(
            "new_session", {"name": "test-pipe"}, _empty_state(), catalog, scratch_dir
        )
        session_id = new_result["data"]["session_id"]

        _dispatch_tool(
            "save_session",
            {"session_id": session_id},
            _empty_state(),
            catalog,
            scratch_dir,
        )

        load_result = _dispatch_tool(
            "load_session",
            {"session_id": session_id},
            _empty_state(),
            catalog,
            scratch_dir,
        )
        assert load_result["success"] is True
        assert load_result["state"]["metadata"]["name"] == "test-pipe"

    def test_generate_yaml_returns_string(
        self, catalog: CatalogService, scratch_dir: Path
    ) -> None:
        result = _dispatch_tool(
            "generate_yaml", {}, _empty_state(), catalog, scratch_dir
        )
        assert result["success"] is True
        assert isinstance(result["data"], str)

    def test_unknown_tool_returns_error(
        self, catalog: CatalogService, scratch_dir: Path
    ) -> None:
        result = _dispatch_tool(
            "not_a_real_tool", {}, _empty_state(), catalog, scratch_dir
        )
        assert result["success"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/composer_mcp/test_server.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_tool_defs'`

- [ ] **Step 3: Write the MCP server**

Create `src/elspeth/composer_mcp/server.py`:

```python
"""MCP server for ELSPETH pipeline composition.

Exposes the web composer's pipeline-building tools (discovery + mutation)
plus session management and YAML generation. No Landscape access,
no blob tools, no secret tools.

Uses the existing execute_tool() dispatcher from web.composer.tools
for all discovery and mutation operations. Session management and
YAML generation are handled locally.

Layer: L3 (application). Imports from L3 (web.composer, web.catalog).

Usage:
    elspeth-composer                          # Uses .scratch/ in cwd
    elspeth-composer --scratch-dir /tmp/my    # Custom scratch dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from elspeth.composer_mcp.session import SessionManager, SessionNotFoundError
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import (
    ToolResult,
    _DISCOVERY_TOOLS,
    _MUTATION_TOOLS,
    execute_tool,
    get_tool_definitions,
)
from elspeth.web.composer.yaml_generator import generate_yaml

__all__ = ["create_server", "main"]

logger = logging.getLogger(__name__)


# --- Session and utility tool names (not in web composer) ---

_SESSION_TOOL_NAMES = frozenset({
    "new_session",
    "save_session",
    "load_session",
    "list_sessions",
    "delete_session",
    "generate_yaml",
})

# Web composer tools we expose (discovery + mutation, no blob/secret)
_COMPOSER_TOOL_NAMES = frozenset(_DISCOVERY_TOOLS) | frozenset(_MUTATION_TOOLS)


def _build_tool_defs() -> list[Tool]:
    """Build MCP Tool definitions from web composer + session tools.

    Filters the web composer's 27 tools down to the 21 discovery/mutation
    tools (no blob, no secret), then adds 6 session/utility tools.
    """
    tools: list[Tool] = []

    # Import web composer tool definitions and filter to discovery + mutation
    for defn in get_tool_definitions():
        if defn["name"] in _COMPOSER_TOOL_NAMES:
            tools.append(
                Tool(
                    name=defn["name"],
                    description=defn["description"],
                    inputSchema={
                        "type": "object",
                        "properties": defn["parameters"].get("properties", {}),
                        "required": defn["parameters"].get("required", []),
                    },
                )
            )

    # Session management tools
    tools.append(
        Tool(
            name="new_session",
            description="Create a new empty pipeline composition session. Returns session ID and empty state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Pipeline name (default: 'Untitled Pipeline').",
                    },
                },
                "required": [],
            },
        )
    )
    tools.append(
        Tool(
            name="save_session",
            description="Save the current composition state to disk as a JSON file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to save under.",
                    },
                },
                "required": ["session_id"],
            },
        )
    )
    tools.append(
        Tool(
            name="load_session",
            description="Load a previously saved composition session from disk. Returns the saved state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to load.",
                    },
                },
                "required": ["session_id"],
            },
        )
    )
    tools.append(
        Tool(
            name="list_sessions",
            description="List all saved composition sessions with name and version.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
    )
    tools.append(
        Tool(
            name="delete_session",
            description="Delete a saved composition session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to delete.",
                    },
                },
                "required": ["session_id"],
            },
        )
    )
    tools.append(
        Tool(
            name="generate_yaml",
            description="Generate ELSPETH pipeline YAML from the current composition state. Returns the YAML string.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
    )

    return tools


def _dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceImpl,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Dispatch a tool call and return a serializable result dict.

    For discovery/mutation tools, delegates to web.composer.tools.execute_tool().
    For session/utility tools, handles locally.

    Returns a dict with keys: success, state (serialized), validation,
    affected_nodes, version, data (optional).
    """
    # Session/utility tools
    if tool_name == "new_session":
        manager = SessionManager(scratch_dir)
        name = arguments.get("name", "Untitled Pipeline")
        session_id, new_state = manager.new_session(name=name)
        manager.save(session_id, new_state)
        validation = new_state.validate()
        return {
            "success": True,
            "state": new_state.to_dict(),
            "validation": {
                "is_valid": validation.is_valid,
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
                "suggestions": list(validation.suggestions),
            },
            "affected_nodes": [],
            "version": new_state.version,
            "data": {"session_id": session_id},
        }

    if tool_name == "save_session":
        session_id = arguments["session_id"]
        manager = SessionManager(scratch_dir)
        path = manager.save(session_id, state)
        return {
            "success": True,
            "state": state.to_dict(),
            "validation": _validation_dict(state),
            "affected_nodes": [],
            "version": state.version,
            "data": {"path": str(path)},
        }

    if tool_name == "load_session":
        session_id = arguments["session_id"]
        manager = SessionManager(scratch_dir)
        try:
            loaded = manager.load(session_id)
        except SessionNotFoundError as exc:
            return {
                "success": False,
                "state": state.to_dict(),
                "validation": _validation_dict(state),
                "affected_nodes": [],
                "version": state.version,
                "data": {"error": str(exc)},
            }
        return {
            "success": True,
            "state": loaded.to_dict(),
            "validation": _validation_dict(loaded),
            "affected_nodes": [],
            "version": loaded.version,
            "data": {"session_id": session_id},
        }

    if tool_name == "list_sessions":
        manager = SessionManager(scratch_dir)
        sessions = manager.list_sessions()
        return {
            "success": True,
            "state": state.to_dict(),
            "validation": _validation_dict(state),
            "affected_nodes": [],
            "version": state.version,
            "data": sessions,
        }

    if tool_name == "delete_session":
        session_id = arguments["session_id"]
        manager = SessionManager(scratch_dir)
        try:
            manager.delete(session_id)
        except SessionNotFoundError as exc:
            return {
                "success": False,
                "state": state.to_dict(),
                "validation": _validation_dict(state),
                "affected_nodes": [],
                "version": state.version,
                "data": {"error": str(exc)},
            }
        return {
            "success": True,
            "state": state.to_dict(),
            "validation": _validation_dict(state),
            "affected_nodes": [],
            "version": state.version,
            "data": {"deleted": session_id},
        }

    if tool_name == "generate_yaml":
        yaml_str = generate_yaml(state)
        return {
            "success": True,
            "state": state.to_dict(),
            "validation": _validation_dict(state),
            "affected_nodes": [],
            "version": state.version,
            "data": yaml_str,
        }

    # Discovery and mutation tools — delegate to web composer
    if tool_name in _COMPOSER_TOOL_NAMES:
        result: ToolResult = execute_tool(
            tool_name,
            arguments,
            state,
            catalog,
            data_dir=None,
        )
        response = result.to_dict()
        response["state"] = result.updated_state.to_dict()
        return response

    # Unknown tool
    return {
        "success": False,
        "state": state.to_dict(),
        "validation": _validation_dict(state),
        "affected_nodes": [],
        "version": state.version,
        "data": {"error": f"Unknown tool: {tool_name}"},
    }


def _validation_dict(state: CompositionState) -> dict[str, Any]:
    v = state.validate()
    return {
        "is_valid": v.is_valid,
        "errors": list(v.errors),
        "warnings": list(v.warnings),
        "suggestions": list(v.suggestions),
    }


def create_server(scratch_dir: Path) -> Server:
    """Create and configure the MCP server instance."""
    server = Server("elspeth-composer")

    catalog = CatalogServiceImpl(get_shared_plugin_manager())
    tool_defs = _build_tool_defs()

    # Mutable state ref — updated after each mutation tool call.
    # The MCP server is single-client (stdio), so no concurrency concern.
    state_ref: list[CompositionState] = [
        CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=0,
        )
    ]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tool_defs

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = _dispatch_tool(
            name, arguments or {}, state_ref[0], catalog, scratch_dir
        )

        # Update server-side state if the tool returned a new state
        if result.get("success") and "state" in result:
            state_ref[0] = CompositionState.from_dict(result["state"])

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


async def run_server(scratch_dir: Path) -> None:
    """Run the MCP server with stdio transport."""
    server = create_server(scratch_dir)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ELSPETH Composer MCP Server - Pipeline building tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Uses .scratch/ in current directory
    elspeth-composer

    # Custom scratch directory
    elspeth-composer --scratch-dir /tmp/my-pipelines
""",
    )
    parser.add_argument(
        "--scratch-dir",
        default=".scratch",
        help="Directory for session persistence (default: .scratch/)",
    )

    args = parser.parse_args()

    import asyncio

    asyncio.run(run_server(Path(args.scratch_dir)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/composer_mcp/test_server.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/composer_mcp/server.py tests/unit/composer_mcp/test_server.py
git commit -m "feat(composer-mcp): add MCP server with tool dispatch and session management"
```

---

### Task 4: Wire Entry Point and Integration Test

**Files:**
- Modify: `pyproject.toml` (line ~210, `[project.scripts]`)

- [ ] **Step 1: Add console script entry point**

In `pyproject.toml`, after the `elspeth-mcp` line, add:

```toml
elspeth-composer = "elspeth.composer_mcp:main"
```

- [ ] **Step 2: Reinstall package to register new entry point**

```bash
uv pip install -e ".[dev,mcp]"
```

- [ ] **Step 3: Verify entry point exists**

```bash
elspeth-composer --help
```

Expected output includes "ELSPETH Composer MCP Server" and `--scratch-dir` flag.

- [ ] **Step 4: Run full test suite for composer_mcp**

```bash
.venv/bin/python -m pytest tests/unit/composer_mcp/ -v
```

Expected: All tests pass.

- [ ] **Step 5: Run tier model enforcer to check layer imports**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: No violations (composer_mcp is L3, imports from L0 and L3 only).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "feat(composer-mcp): wire elspeth-composer CLI entry point"
```

---

### Task 5: MCP Configuration for Claude Code

**Files:**
- Modify: `.claude/settings.json` or equivalent MCP config

This task wires `elspeth-composer` into Claude Code's MCP server list so the tools appear in conversation.

- [ ] **Step 1: Check current MCP configuration**

Look at `.claude/settings.json` or `.mcp.json` for how `elspeth-mcp` is configured. The composer server should follow the same pattern.

- [ ] **Step 2: Add composer MCP server config**

Add to the MCP servers config:

```json
{
  "elspeth-composer": {
    "command": "elspeth-composer",
    "args": ["--scratch-dir", ".scratch"]
  }
}
```

- [ ] **Step 3: Verify tools appear**

Start a new Claude Code session and verify the composer tools are available (list_sources, set_source, new_session, etc.).

- [ ] **Step 4: Smoke test — build a simple pipeline**

In conversation, try:
1. Call `new_session` with name "smoke-test"
2. Call `list_sources` to see available plugins
3. Call `set_source` with csv plugin
4. Call `set_output` with csv sink
5. Call `upsert_edge` to connect source to sink
6. Call `generate_yaml` to see the output
7. Call `save_session`

Verify each step returns valid results and the final YAML is a valid ELSPETH pipeline.

- [ ] **Step 5: Commit**

```bash
git add .claude/settings.json  # or .mcp.json
git commit -m "chore(composer-mcp): add MCP config for Claude Code"
```

---

## Summary

| Task | What it builds | Test count |
|------|---------------|------------|
| 1 | `.scratch/` folder + `.gitignore` | 0 (infra) |
| 2 | Session persistence (`SessionManager`) | 9 tests |
| 3 | MCP server with tool dispatch | 12 tests |
| 4 | CLI entry point + integration check | 0 (wiring) |
| 5 | Claude Code MCP config + smoke test | 0 (manual) |

**Total new files:** 5 source + 3 test
**Total modified files:** 2 (`.gitignore`, `pyproject.toml`) + MCP config
**Estimated tools exposed:** 27 (21 composer + 6 session/utility)
