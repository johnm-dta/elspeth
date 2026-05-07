# Web UX — LLM Composer MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular monolith web application for chat-first pipeline authoring with server-side LLM composition, dry-run validation, and execution with WebSocket progress.

**Architecture:** Single FastAPI process with five internal modules (auth, catalog, composer, execution, sessions), each behind a Python Protocol interface. React/TypeScript SPA frontend. Pipeline execution in background threads. Protocol boundaries designed for later extraction to microservices.

**Tech Stack:** FastAPI, SQLAlchemy (SQLite dev / Postgres prod), LiteLLM, React 18, TypeScript, Vite, React Flow, Zustand, WebSocket.

**Spec:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`

---

## Review Findings Applied

This plan incorporates fixes for 8 blocking issues identified by a 5-reviewer
panel (architecture, systems, Python, quality, UX). Key changes:

- **B1/B2/B7:** Execution service uses `loop.call_soon_threadsafe()` for
  async/thread bridge, always passes `shutdown_event` to Orchestrator (avoids
  signal handler crash), wraps `_run_pipeline()` in `try/BaseException/finally`
  with `future.add_done_callback()`.
- **B3:** `WebSettings` includes `landscape_url` and `payload_store_path` for
  Orchestrator construction. Defaults to `{data_dir}/runs/audit.db` and
  `{data_dir}/payloads/`.
- **B4:** New Task 0.1 extracts `_get_plugin_manager()` from `cli.py` to
  `plugins/infrastructure/manager.py` before web code is written.
- **B5:** Upload path sanitizes `user_id` with `Path(user_id).name`. Test for
  path traversal included.
- **B6:** `runs` table has a partial unique constraint enforcing one active run
  per session. Application-level check-and-set for SQLite compatibility.
- **B8:** All async test functions include `@pytest.mark.asyncio`.

### Outstanding Warnings (not blocking — address during implementation)

| # | Warning | Source | Notes |
|---|---------|--------|-------|
| W1 | Composing indicator needs turn-level progress (SSE or polling) | UX | Static dots insufficient for 10-30s waits. Add `/api/sessions/{id}/compose/status` |
| W2 | No version history / undo UI in frontend | UX | Backend has versioned state; frontend needs a revert affordance |
| W3 | Stale validation after composition change | UX | New state version must clear validation results and disable Execute |
| W4 | Spec view keyboard accessibility | UX | Component cards must be focusable via Tab, selectable via Enter/Space |
| W5 | IDOR on session routes | QUALITY | `GET /api/sessions/{id}` must verify user ownership, test for foreign session |
| W6 | No Alembic migration for web DB | ARCH | Call `metadata.create_all()` on startup for dev; add Alembic for production |
| W7 | CLI settings not forwarded through uvicorn factory string | PYTHON | Use env vars or pass app instance directly when `--reload=False` |
| W8 | No file size limit on upload | QUALITY | Add `max_upload_bytes` to WebSettings (default 100MB) |
| W9 | Empty states not designed in frontend | UX | Welcome prompts, placeholder text in all inspector tabs |
| W10 | Multiple uvicorn workers break WebSocket progress | SYSTEMS | Add startup warning if `WEB_CONCURRENCY > 1` |
| W11 | CompositionState JSON schema evolution | ARCH | Add `_version: int` to JSON envelope for forward compatibility |
| W12 | HTTP timeout for composer loop | ARCH | Add `composer_timeout_seconds` to WebSettings |
| W13 | Outer LLM conversation loop unbounded in cost | SYSTEMS | No per-session rate limit; add before non-single-user deployment |
| W14 | WebSocket reconnect loses prior events | SYSTEMS | Known v1 limitation; document; Redis Streams later |
| W15 | SessionService dual-duty will require split on extraction | ARCH | CompositionState → Artifact Service; Session → Task DB |
| W16 | `secret_key` default unsafe for production | QUALITY | Add startup guard rejecting default outside test envs |
| W17 | No frontend tests | QUALITY | Add Vitest + React Testing Library for hooks + SpecView linking |
| W18 | `validate_pipeline()` must catch only typed exceptions | PYTHON | Use specific exception types, not bare `except Exception` |

---

## Scope Decomposition

This plan is structured as 6 phases, each producing working, testable software:

| Phase | What It Builds | Depends On |
|-------|---------------|------------|
| 0. Pre-work | Extract plugin manager singleton from cli.py to plugins/infrastructure/ | Nothing |
| 1. Foundation | Package structure, config, app factory, `[webui]` extra, `elspeth web` CLI | Phase 0 |
| 2. Auth & Sessions | AuthProvider protocol, LocalAuthProvider, session/message persistence, CRUD API | Phase 1 |
| 3. Catalog | CatalogService wrapping PluginManager, catalog REST API | Phase 1 |
| 4. Composer | CompositionState model, LLM tool-use loop, composition tools, chat API | Phases 2, 3 |
| 5. Execution | Dry-run validation, background execution, WebSocket progress, run history | Phases 2, 4 |
| 6. Frontend | React SPA — chat panel, inspector (spec/graph/YAML/runs), auth, progress | Phases 2-5 |

---

## File Map

### Phase 0: Pre-work

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/elspeth/plugins/infrastructure/manager.py` | Add `get_shared_plugin_manager()` singleton |
| Modify | `src/elspeth/cli_helpers.py` | Use `get_shared_plugin_manager()` from manager.py instead of `cli._get_plugin_manager()` |
| Modify | `src/elspeth/cli.py` | Remove `_get_plugin_manager()` / `_plugin_manager_cache`, delegate to manager.py |
| Create | `tests/unit/plugins/test_manager_singleton.py` | Test shared singleton |

### Phase 1: Foundation

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/__init__.py` | Package init |
| Create | `src/elspeth/web/app.py` | FastAPI application factory (`create_app()`) |
| Create | `src/elspeth/web/config.py` | `WebSettings` — port, auth provider, LLM model, CORS origins, data dir |
| Create | `src/elspeth/web/dependencies.py` | FastAPI `Depends()` providers for all services |
| Modify | `pyproject.toml` | Add `[webui]` optional extra |
| Modify | `src/elspeth/cli.py` | Add `elspeth web` subcommand |
| Create | `tests/unit/web/__init__.py` | Test package |
| Create | `tests/unit/web/test_app.py` | App factory smoke tests |

### Phase 2: Auth & Sessions

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/auth/__init__.py` | Module init |
| Create | `src/elspeth/web/auth/protocol.py` | `AuthProvider` protocol |
| Create | `src/elspeth/web/auth/models.py` | `UserIdentity`, `UserProfile`, `TokenPayload` |
| Create | `src/elspeth/web/auth/local.py` | `LocalAuthProvider` — password hashing, JWT |
| Create | `src/elspeth/web/auth/oidc.py` | `OIDCAuthProvider` — JWKS discovery, token validation |
| Create | `src/elspeth/web/auth/entra.py` | `EntraAuthProvider` — Entra-specific tenant/group claims |
| Create | `src/elspeth/web/auth/middleware.py` | FastAPI dependency that extracts `UserIdentity` from request |
| Create | `src/elspeth/web/auth/routes.py` | `/api/auth/login`, `/api/auth/token`, `/api/auth/me` |
| Create | `src/elspeth/web/sessions/__init__.py` | Module init |
| Create | `src/elspeth/web/sessions/protocol.py` | `SessionService` protocol |
| Create | `src/elspeth/web/sessions/models.py` | SQLAlchemy models: `Session`, `ChatMessage`, `CompositionStateRecord`, `Run`, `RunEvent` |
| Create | `src/elspeth/web/sessions/service.py` | `SessionServiceImpl` — CRUD, state versioning |
| Create | `src/elspeth/web/sessions/routes.py` | `/api/sessions/*` endpoints |
| Create | `src/elspeth/web/sessions/schemas.py` | Pydantic request/response models |
| Create | `tests/unit/web/auth/test_local_provider.py` | LocalAuthProvider tests |
| Create | `tests/unit/web/auth/test_middleware.py` | Auth middleware tests |
| Create | `tests/unit/web/sessions/test_service.py` | SessionService CRUD tests |
| Create | `tests/unit/web/sessions/test_routes.py` | Session API endpoint tests |

### Phase 3: Catalog

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/catalog/__init__.py` | Module init |
| Create | `src/elspeth/web/catalog/protocol.py` | `CatalogService` protocol |
| Create | `src/elspeth/web/catalog/service.py` | `CatalogServiceImpl` — wraps `PluginManager` |
| Create | `src/elspeth/web/catalog/routes.py` | `/api/catalog/*` endpoints |
| Create | `src/elspeth/web/catalog/schemas.py` | `PluginSummary`, `PluginSchemaInfo` response models |
| Create | `tests/unit/web/catalog/test_service.py` | CatalogService tests (plugin discovery serialization) |
| Create | `tests/unit/web/catalog/test_routes.py` | Catalog API endpoint tests |

### Phase 4: Composer

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/composer/__init__.py` | Module init |
| Create | `src/elspeth/web/composer/protocol.py` | `ComposerService` protocol |
| Create | `src/elspeth/web/composer/state.py` | `CompositionState`, `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec` |
| Create | `src/elspeth/web/composer/tools.py` | Tool definitions: discovery + mutation tools |
| Create | `src/elspeth/web/composer/prompts.py` | System prompt template, context injection |
| Create | `src/elspeth/web/composer/service.py` | `ComposerServiceImpl` — LLM tool-use loop |
| Create | `src/elspeth/web/composer/yaml_generator.py` | `CompositionState` → ELSPETH pipeline YAML |
| Create | `tests/unit/web/composer/test_state.py` | CompositionState immutability, versioning |
| Create | `tests/unit/web/composer/test_tools.py` | Tool execution: validation, state reflection |
| Create | `tests/unit/web/composer/test_yaml_generator.py` | YAML generation correctness |
| Create | `tests/unit/web/composer/test_service.py` | Composer loop with mocked LLM |

### Phase 5: Execution

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/execution/__init__.py` | Module init |
| Create | `src/elspeth/web/execution/protocol.py` | `ExecutionService` protocol |
| Create | `src/elspeth/web/execution/service.py` | `ExecutionServiceImpl` — background thread orchestration |
| Create | `src/elspeth/web/execution/validation.py` | Dry-run: YAML → `ExecutionGraph` → `graph.validate()` |
| Create | `src/elspeth/web/execution/progress.py` | `ProgressBroadcaster` — in-process event forwarding to WebSocket |
| Create | `src/elspeth/web/execution/routes.py` | `/api/sessions/{id}/validate`, `/api/sessions/{id}/execute`, `/api/runs/*`, `WS /ws/runs/{id}` |
| Create | `src/elspeth/web/execution/schemas.py` | `ValidationResult`, `RunStatus`, `RunEvent` |
| Create | `tests/unit/web/execution/test_validation.py` | Dry-run validation with valid/invalid pipelines |
| Create | `tests/unit/web/execution/test_progress.py` | WebSocket broadcast tests |
| Create | `tests/integration/web/test_execute_pipeline.py` | End-to-end: compose → validate → execute → results |

### Phase 6: Frontend

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/frontend/package.json` | Dependencies: react, react-dom, reactflow, zustand, typescript |
| Create | `src/elspeth/web/frontend/tsconfig.json` | TypeScript config |
| Create | `src/elspeth/web/frontend/vite.config.ts` | Vite config with API proxy to FastAPI |
| Create | `src/elspeth/web/frontend/index.html` | SPA entry point |
| Create | `src/elspeth/web/frontend/src/main.tsx` | React root |
| Create | `src/elspeth/web/frontend/src/App.tsx` | Top-level layout: sidebar + chat + inspector |
| Create | `src/elspeth/web/frontend/src/api/client.ts` | Typed API client (generated from OpenAPI or hand-written) |
| Create | `src/elspeth/web/frontend/src/api/websocket.ts` | WebSocket connection manager |
| Create | `src/elspeth/web/frontend/src/stores/authStore.ts` | Auth state (Zustand) |
| Create | `src/elspeth/web/frontend/src/stores/sessionStore.ts` | Session + composition state (Zustand) |
| Create | `src/elspeth/web/frontend/src/stores/executionStore.ts` | Run state + WebSocket events (Zustand) |
| Create | `src/elspeth/web/frontend/src/components/common/Layout.tsx` | Three-panel layout shell |
| Create | `src/elspeth/web/frontend/src/components/common/AuthGuard.tsx` | Auth redirect wrapper |
| Create | `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx` | Session list, new session |
| Create | `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` | Message list + input |
| Create | `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx` | Single message display |
| Create | `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx` | Input + send button + composing indicator |
| Create | `src/elspeth/web/frontend/src/components/chat/ComposingIndicator.tsx` | Typing dots animation |
| Create | `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` | Tab container (Spec, Graph, YAML, Runs) |
| Create | `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx` | Component cards with click-to-highlight linking |
| Create | `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx` | React Flow DAG (read-only) |
| Create | `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx` | Read-only YAML with copy |
| Create | `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx` | Run history list |
| Create | `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx` | Live row count + exceptions |
| Create | `src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx` | Pass/fail with per-component errors |
| Create | `src/elspeth/web/frontend/src/hooks/useSession.ts` | Session CRUD hook |
| Create | `src/elspeth/web/frontend/src/hooks/useComposer.ts` | Message send + state update hook |
| Create | `src/elspeth/web/frontend/src/hooks/useWebSocket.ts` | WebSocket connection + event hook |
| Create | `src/elspeth/web/frontend/src/hooks/useAuth.ts` | Auth flow hook |
| Create | `src/elspeth/web/frontend/src/types/index.ts` | Shared TypeScript types |
| Modify | `src/elspeth/web/app.py` | Add static file serving for `frontend/dist/` |

---

## Global Implementation Notes

**B8 — All async tests require `@pytest.mark.asyncio`.** The project uses
pytest-asyncio without `asyncio_mode = "auto"`. Every `async def test_*`
function in every phase must have `@pytest.mark.asyncio` or it will silently
pass without running. The test code blocks in this plan show the marker where
async tests are fully written out; for tasks where test code is described but
not shown, the implementer must add the marker.

**Frozen dataclass convention.** All frozen dataclasses with container fields
(`list`, `dict`, `set`, `Mapping`, `Sequence`) must call `freeze_fields()` in
`__post_init__`. This applies to `CompositionState`, `ToolResult`, all `*Spec`
types, and any other frozen dataclass created in the web package. Enforced by
CI via `enforce_freeze_guards.py`.

**AuthenticationError location.** Define `AuthenticationError` in
`auth/models.py` (not `auth/protocol.py`). Keep `protocol.py` as a pure
structural interface, consistent with the project pattern of exceptions living
in `contracts/errors.py` separate from Protocols.

---

## Phase 0: Pre-work — Extract Plugin Manager Singleton (B4)

### Task 0.1: Move Plugin Manager Singleton to plugins/infrastructure/

**Why:** `instantiate_plugins_from_config()` in `cli_helpers.py` reaches into
`cli.py` for `_get_plugin_manager()`. The web module cannot depend on the CLI
module — both are L3 peers. Move the singleton to `plugins/infrastructure/manager.py`
where `PluginManager` already lives.

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/manager.py`
- Modify: `src/elspeth/cli_helpers.py`
- Modify: `src/elspeth/cli.py`
- Create: `tests/unit/plugins/test_manager_singleton.py`

- [ ] **Step 1: Write singleton test**

```python
# tests/unit/plugins/test_manager_singleton.py
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager


def test_get_shared_plugin_manager_returns_same_instance() -> None:
    pm1 = get_shared_plugin_manager()
    pm2 = get_shared_plugin_manager()
    assert pm1 is pm2


def test_get_shared_plugin_manager_has_builtin_plugins() -> None:
    pm = get_shared_plugin_manager()
    assert len(pm.get_sources()) > 0
    assert len(pm.get_transforms()) > 0
    assert len(pm.get_sinks()) > 0
```

- [ ] **Step 2: Add `get_shared_plugin_manager()` to `manager.py`**

```python
# At the end of src/elspeth/plugins/infrastructure/manager.py
_shared_instance: PluginManager | None = None

def get_shared_plugin_manager() -> PluginManager:
    """Return the shared plugin manager singleton.

    Creates and initialises with builtin plugins on first call.
    Used by both CLI and web entry points.
    """
    global _shared_instance  # noqa: PLW0603
    if _shared_instance is None:
        _shared_instance = PluginManager()
        _shared_instance.register_builtin_plugins()
    return _shared_instance
```

- [ ] **Step 3: Update `cli_helpers.py` to use the new function**

Replace the import of `_get_plugin_manager` from `cli` with
`get_shared_plugin_manager` from `plugins.infrastructure.manager`.

- [ ] **Step 4: Remove `_get_plugin_manager()` and `_plugin_manager_cache` from `cli.py`**

Update any remaining references in `cli.py` to use `get_shared_plugin_manager()`.

- [ ] **Step 5: Run full test suite, commit**

```bash
.venv/bin/python -m pytest tests/ -x
git commit -m "refactor: extract plugin manager singleton from cli.py to manager.py"
```

---

## Phase 1: Foundation

### Task 1.1: Package Structure and Config

**Files:**
- Create: `src/elspeth/web/__init__.py`
- Create: `src/elspeth/web/config.py`
- Create: `tests/unit/web/__init__.py`

- [ ] **Step 1: Write config tests**

```python
# tests/unit/web/test_config.py
import pytest
from elspeth.web.config import WebSettings


def test_web_settings_defaults() -> None:
    settings = WebSettings()
    assert settings.port == 8000
    assert settings.host == "127.0.0.1"
    assert settings.auth_provider == "local"
    assert settings.cors_origins == ["http://localhost:5173"]
    assert settings.data_dir.name == "data"


def test_web_settings_custom_values() -> None:
    settings = WebSettings(
        port=9090,
        host="0.0.0.0",
        auth_provider="entra",
        cors_origins=["https://app.example.com"],
    )
    assert settings.port == 9090
    assert settings.auth_provider == "entra"


def test_web_settings_invalid_auth_provider() -> None:
    with pytest.raises(ValueError, match="auth_provider"):
        WebSettings(auth_provider="invalid")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web'`

- [ ] **Step 3: Implement WebSettings**

```python
# src/elspeth/web/__init__.py
"""ELSPETH Web UX — LLM Composer MVP."""

# src/elspeth/web/config.py
"""Web application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


class WebSettings(BaseModel):
    """Configuration for the ELSPETH web application."""

    host: str = "127.0.0.1"
    port: int = 8000
    auth_provider: Literal["local", "oidc", "entra"] = "local"
    cors_origins: list[str] = ["http://localhost:5173"]
    data_dir: Path = Path("data")
    composer_model: str = "gpt-4o"
    composer_max_turns: int = 20
    composer_timeout_seconds: float = 120.0
    secret_key: str = "change-me-in-production"
    max_upload_bytes: int = 100 * 1024 * 1024  # 100MB

    # Execution infrastructure (B3 fix)
    # Defaults derive from data_dir if not explicitly set
    landscape_url: str | None = None  # default: sqlite:///{data_dir}/runs/audit.db
    payload_store_path: Path | None = None  # default: {data_dir}/payloads/

    # OIDC/Entra-specific (optional)
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    entra_tenant_id: str | None = None

    # Note: auth_provider uses Literal type which handles validation
    # automatically — no manual @field_validator needed.

    def get_landscape_url(self) -> str:
        """Resolve landscape DB URL, defaulting to data_dir-relative path."""
        if self.landscape_url is not None:
            return self.landscape_url
        db_path = self.data_dir / "runs" / "audit.db"
        return f"sqlite:///{db_path}"

    def get_payload_store_path(self) -> Path:
        """Resolve payload store path, defaulting to data_dir-relative path."""
        if self.payload_store_path is not None:
            return self.payload_store_path
        return self.data_dir / "payloads"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/ tests/unit/web/
git commit -m "feat(web): add web package with WebSettings config"
```

### Task 1.2: FastAPI Application Factory

**Files:**
- Create: `src/elspeth/web/app.py`
- Create: `src/elspeth/web/dependencies.py`
- Create: `tests/unit/web/test_app.py`

- [ ] **Step 1: Write app factory tests**

```python
# tests/unit/web/test_app.py
from starlette.testclient import TestClient

from elspeth.web.app import create_app
from elspeth.web.config import WebSettings


def test_create_app_returns_fastapi_instance() -> None:
    settings = WebSettings()
    app = create_app(settings)
    assert app.title == "ELSPETH Web"


def test_health_endpoint() -> None:
    settings = WebSettings()
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Implement app factory**

```python
# src/elspeth/web/app.py
"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from elspeth.web.config import WebSettings


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = WebSettings()

    app = FastAPI(title="ELSPETH Web", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
```

```python
# src/elspeth/web/dependencies.py
"""FastAPI dependency injection for service instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request

if TYPE_CHECKING:
    from elspeth.web.auth.protocol import AuthProvider
    from elspeth.web.catalog.protocol import CatalogServiceProtocol
    from elspeth.web.composer.protocol import ComposerServiceProtocol
    from elspeth.web.config import WebSettings
    from elspeth.web.execution.protocol import ExecutionServiceProtocol
    from elspeth.web.sessions.protocol import SessionServiceProtocol


def get_settings(request: Request) -> WebSettings:
    """Get application settings from app state."""
    return request.app.state.settings


# Service dependency stubs — implemented as each phase lands
# Each returns the service instance from app.state, set during startup
```

- [ ] **Step 4: Run tests — verify they pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(web): add FastAPI app factory with health endpoint"
```

### Task 1.3: pyproject.toml Extra and CLI Entry Point

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/elspeth/cli.py`

- [ ] **Step 1: Add `[webui]` extra to pyproject.toml**

Add to `[project.optional-dependencies]`:

```toml
webui = [
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.34,<1",
    "PyJWT[crypto]>=2.8,<3",
    "python-multipart>=0.0.20",
    "websockets>=14.0,<15",
    "httpx>=0.27,<1",
]
```

Update the `all` extra to include `"elspeth[webui]"`.

- [ ] **Step 2: Install the new extra**

```bash
uv pip install -e ".[webui,dev]"
```

- [ ] **Step 3: Add `elspeth web` CLI subcommand**

In `src/elspeth/cli.py`, add alongside the existing `plugins_app` pattern:

```python
@app.command()
def web(
    port: int = typer.Option(8000, help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    auth: str = typer.Option("local", help="Auth provider: local, oidc, entra"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Start the ELSPETH web application."""
    try:
        import uvicorn
        from elspeth.web.config import WebSettings
    except ImportError:
        _console.print(
            "[red]Web UI requires the [webui] extra. "
            "Install with: uv pip install -e '.[webui]'[/]"
        )
        raise typer.Exit(1) from None

    settings = WebSettings(port=port, host=host, auth_provider=auth)
    uvicorn.run(
        "elspeth.web.app:create_app",
        host=settings.host,
        port=settings.port,
        reload=reload,
        factory=True,
    )
```

- [ ] **Step 4: Verify CLI**

```bash
elspeth web --help
```

Expected: Help text showing port, host, auth, reload options.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(web): add [webui] extra and elspeth web CLI command"
```

---

## Phase 2: Auth & Sessions

### Task 2.1: AuthProvider Protocol and Models

**Files:**
- Create: `src/elspeth/web/auth/__init__.py`
- Create: `src/elspeth/web/auth/protocol.py`
- Create: `src/elspeth/web/auth/models.py`

- [ ] **Step 1: Define protocol and models**

```python
# src/elspeth/web/auth/protocol.py
"""Authentication provider protocol."""

from __future__ import annotations

from typing import Protocol

from elspeth.web.auth.models import UserIdentity, UserProfile


class AuthProvider(Protocol):
    """Protocol for pluggable authentication providers."""

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate a token and return the authenticated identity.

        Raises AuthenticationError if the token is invalid.
        """
        ...

    async def get_user_info(self, token: str) -> UserProfile:
        """Get full user profile from a valid token."""
        ...


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    def __init__(self, detail: str = "Authentication failed") -> None:
        self.detail = detail
        super().__init__(detail)
```

```python
# src/elspeth/web/auth/models.py
"""Authentication data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Minimal authenticated identity — returned from every auth check."""

    user_id: str
    username: str


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Extended user profile information."""

    user_id: str
    username: str
    display_name: str
    email: str | None = None
    groups: tuple[str, ...] = ()
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(web/auth): add AuthProvider protocol and identity models"
```

### Task 2.2: LocalAuthProvider

**Files:**
- Create: `src/elspeth/web/auth/local.py`
- Create: `tests/unit/web/auth/test_local_provider.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/auth/test_local_provider.py
import pytest

from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.protocol import AuthenticationError


@pytest.fixture
def provider(tmp_path):
    return LocalAuthProvider(
        db_path=tmp_path / "auth.db",
        secret_key="test-secret-key",
    )


@pytest.mark.asyncio
async def test_create_user_and_login(provider):
    provider.create_user("alice", "password123", display_name="Alice")
    token = provider.login("alice", "password123")
    identity = await provider.authenticate(token)
    assert identity.user_id == "alice"
    assert identity.username == "alice"


@pytest.mark.asyncio
async def test_login_wrong_password(provider):
    provider.create_user("alice", "password123")
    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        provider.login("alice", "wrong")


@pytest.mark.asyncio
async def test_authenticate_invalid_token(provider):
    with pytest.raises(AuthenticationError, match="Invalid token"):
        await provider.authenticate("garbage-token")


@pytest.mark.asyncio
async def test_get_user_info(provider):
    provider.create_user("alice", "pw", display_name="Alice Smith", email="a@b.com")
    token = provider.login("alice", "pw")
    profile = await provider.get_user_info(token)
    assert profile.display_name == "Alice Smith"
    assert profile.email == "a@b.com"
```

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Implement LocalAuthProvider**

LocalAuthProvider uses SQLite for user storage, bcrypt for password
hashing, and PyJWT for JWT tokens. The `login()` method returns a JWT.
`authenticate()` validates the JWT and returns `UserIdentity`.
`get_user_info()` returns the full `UserProfile`.

Key implementation details:
- SQLite database created at `db_path` on first use
- Users table: `(user_id TEXT PK, password_hash TEXT, display_name TEXT, email TEXT)`
- JWT payload: `{"sub": user_id, "username": username, "exp": expiry}`
- Token expiry: 24 hours (configurable)

- [ ] **Step 4: Run tests — verify they pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(web/auth): implement LocalAuthProvider with JWT"
```

### Task 2.3: OIDC and Entra Auth Providers

**Files:**
- Create: `src/elspeth/web/auth/oidc.py`
- Create: `src/elspeth/web/auth/entra.py`
- Create: `tests/unit/web/auth/test_oidc_provider.py`
- Create: `tests/unit/web/auth/test_entra_provider.py`

- [ ] **Step 1: Write tests for OIDCAuthProvider**

Test JWKS discovery (mocked with httpx), token validation with valid/invalid/expired
tokens, audience validation. Use `PyJWT` to create test JWTs signed with
a known RSA key.

- [ ] **Step 2: Implement OIDCAuthProvider**

Uses httpx to fetch JWKS from `{issuer}/.well-known/openid-configuration`.
Validates JWT signature, expiry, audience, issuer. Caches JWKS with TTL.

- [ ] **Step 3: Write tests for EntraAuthProvider**

Extends OIDC with tenant validation (`tid` claim) and group extraction
(`groups` claim). Tests: valid Entra token, wrong tenant, group extraction.

- [ ] **Step 4: Implement EntraAuthProvider**

Subclasses or wraps `OIDCAuthProvider`. Adds tenant ID validation and maps
Entra group claims to `UserProfile.groups`.

- [ ] **Step 5: Run all auth tests, commit**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/ -v
git commit -m "feat(web/auth): add OIDC and Entra auth providers"
```

### Task 2.4: Auth Middleware and Routes

**Files:**
- Create: `src/elspeth/web/auth/middleware.py`
- Create: `src/elspeth/web/auth/routes.py`
- Create: `tests/unit/web/auth/test_middleware.py`

- [ ] **Step 1: Write middleware tests**

Test: request with valid Bearer token → `UserIdentity` injected into request
state. Request with no token → 401. Request with invalid token → 401.

- [ ] **Step 2: Implement auth middleware**

FastAPI dependency that reads `Authorization: Bearer <token>`, calls
`auth_provider.authenticate(token)`, returns `UserIdentity`. Raises
`HTTPException(401)` on `AuthenticationError`.

- [ ] **Step 3: Implement auth routes**

`POST /api/auth/login` — body `{username, password}`, calls
`local_provider.login()`, returns `{access_token, token_type}`.
Only available when `auth_provider == "local"`.

`POST /api/auth/token` — refresh (re-issue JWT from existing valid token).

`GET /api/auth/me` — returns `UserProfile` for current token.

- [ ] **Step 4: Register routes in app factory**

- [ ] **Step 5: Run all auth tests, commit**

```bash
git commit -m "feat(web/auth): add auth middleware and login routes"
```

### Task 2.5: SessionService — Models and Persistence

**Files:**
- Create: `src/elspeth/web/sessions/__init__.py`
- Create: `src/elspeth/web/sessions/protocol.py`
- Create: `src/elspeth/web/sessions/models.py`
- Create: `src/elspeth/web/sessions/service.py`
- Create: `tests/unit/web/sessions/test_service.py`

- [ ] **Step 1: Write SessionService tests**

Test session CRUD: create, get, list (user-scoped), archive. Test message
persistence: add message, get history. Test composition state versioning:
create state, new version increments, old versions retrievable.

- [ ] **Step 2: Implement SQLAlchemy models**

Tables: `sessions`, `chat_messages`, `composition_states`, `runs`, `run_events`.
Use SQLAlchemy Core (consistent with Landscape). SQLite for dev.

Key constraints:
- `composition_states` has a unique constraint on `(session_id, version)`.
  Versions are monotonically increasing per session.
- `runs` has an application-level check-and-set enforcing **one active run per
  session** (B6 fix). Before inserting a `pending` run, query for existing
  `pending` or `running` runs on the same session. If found, raise
  `RunAlreadyActiveError`. For Postgres deployments, add a partial unique index:
  `CREATE UNIQUE INDEX uq_runs_active ON runs(session_id) WHERE status IN ('pending', 'running')`.

Schema creation: call `metadata.create_all(engine)` on app startup. For
production Postgres, use Alembic migrations (see W6).

- [ ] **Step 3: Implement SessionServiceImpl**

Implements `SessionServiceProtocol`. Methods:
- `create_session(user_id, title) -> Session`
- `get_session(session_id) -> Session`
- `list_sessions(user_id) -> list[Session]`
- `archive_session(session_id) -> None`
- `add_message(session_id, role, content, tool_calls) -> ChatMessage`
- `get_messages(session_id) -> list[ChatMessage]`
- `save_composition_state(session_id, state) -> CompositionStateRecord`
- `get_current_state(session_id) -> CompositionStateRecord | None`
- `get_state_versions(session_id) -> list[CompositionStateRecord]`

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat(web/sessions): add SessionService with SQLAlchemy persistence"
```

### Task 2.6: Session API Routes

**Files:**
- Create: `src/elspeth/web/sessions/routes.py`
- Create: `src/elspeth/web/sessions/schemas.py`
- Create: `tests/unit/web/sessions/test_routes.py`

- [ ] **Step 1: Write route tests**

Test all endpoints with `TestClient`. Auth required (mock middleware).
Session list is user-scoped. Upload endpoint accepts multipart file,
saves to `{data_dir}/uploads/{sanitized_user_id}/`, returns server path.

**B5 fix — path traversal test:** Include a test where `user_id` contains
`../../etc` and verify the upload is rejected or sanitized to a safe directory
name. The route must use `Path(user_id).name` to strip directory components
before constructing the upload path. Also test file size enforcement against
`WebSettings.max_upload_bytes`.

- [ ] **Step 2: Implement Pydantic schemas**

Request/response models for all session endpoints.

- [ ] **Step 3: Implement routes**

All session CRUD endpoints. File upload endpoint stores to user scratch dir.

- [ ] **Step 4: Register routes in app factory**

- [ ] **Step 5: Run all session tests, commit**

```bash
git commit -m "feat(web/sessions): add session API routes with file upload"
```

---

## Phase 3: Catalog

### Task 3.1: CatalogService

**Files:**
- Create: `src/elspeth/web/catalog/__init__.py`
- Create: `src/elspeth/web/catalog/protocol.py`
- Create: `src/elspeth/web/catalog/service.py`
- Create: `src/elspeth/web/catalog/schemas.py`
- Create: `tests/unit/web/catalog/test_service.py`

- [ ] **Step 1: Write CatalogService tests**

Test: list sources returns all registered source plugins with name, description,
and config schema summary. Same for transforms and sinks. Test `get_schema()`
returns the full Pydantic JSON schema for a named plugin.

Use the existing `PluginManager` with `register_builtin_plugins()` to get
real plugin data in tests.

- [ ] **Step 2: Implement protocol and service**

`CatalogServiceProtocol`:
- `list_sources() -> list[PluginSummary]`
- `list_transforms() -> list[PluginSummary]`
- `list_sinks() -> list[PluginSummary]`
- `get_schema(plugin_type, name) -> PluginSchemaInfo`

`CatalogServiceImpl` wraps `PluginManager`. On init, calls
`register_builtin_plugins()` and caches the discovered plugins.
Serializes plugin classes to `PluginSummary` (name, description,
config fields with types).

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(web/catalog): add CatalogService wrapping PluginManager"
```

### Task 3.2: Catalog API Routes

**Files:**
- Create: `src/elspeth/web/catalog/routes.py`
- Create: `tests/unit/web/catalog/test_routes.py`

- [ ] **Step 1: Write route tests**

Test all catalog endpoints return JSON with correct structure.
Test unknown plugin name returns 404.

- [ ] **Step 2: Implement routes**

- [ ] **Step 3: Register in app factory, run tests, commit**

```bash
git commit -m "feat(web/catalog): add catalog API routes"
```

---

## Phase 4: Composer

### Task 4.1: CompositionState Model

**Files:**
- Create: `src/elspeth/web/composer/__init__.py`
- Create: `src/elspeth/web/composer/state.py`
- Create: `tests/unit/web/composer/test_state.py`

- [ ] **Step 1: Write CompositionState tests**

Test: create empty state, add source, add nodes, add edges. Immutability —
mutations return new instances. Validation: source required, sink required,
edge references valid nodes. Version incrementing.

- [ ] **Step 2: Implement CompositionState**

Frozen dataclasses: `CompositionState`, `SourceSpec`, `NodeSpec`, `EdgeSpec`,
`OutputSpec`, `PipelineMetadata`. Each mutation method returns a new instance
with incremented version. `validate()` returns `(is_valid, errors)`.

Uses `freeze_fields()` for any mutable container fields, consistent with
ELSPETH's frozen dataclass contract.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(web/composer): add CompositionState model"
```

### Task 4.2: Composition Tools

**Files:**
- Create: `src/elspeth/web/composer/tools.py`
- Create: `tests/unit/web/composer/test_tools.py`

- [ ] **Step 1: Write tool tests**

Test each tool: `set_source()` updates state and validates against catalog.
`upsert_node()` adds or updates. `remove_node()` cascades edge removal.
Each tool returns `ToolResult` with full updated state and validation.
Test validation errors (unknown plugin, invalid config) are returned in
the result, not thrown.

- [ ] **Step 2: Implement tools**

Each tool function takes current `CompositionState` and a `CatalogServiceProtocol`,
applies the mutation, validates, returns `ToolResult`. Tool definitions include
JSON Schema descriptions for the LLM.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(web/composer): add composition tools with validation"
```

### Task 4.3: YAML Generator

**Files:**
- Create: `src/elspeth/web/composer/yaml_generator.py`
- Create: `tests/unit/web/composer/test_yaml_generator.py`

- [ ] **Step 1: Write YAML generation tests**

Test: linear pipeline (source → transform → sink) generates valid ELSPETH YAML.
Test: pipeline with gate and multiple sinks. Test: generated YAML can be parsed
by `load_settings()` successfully.

- [ ] **Step 2: Implement YAML generator**

`generate_yaml(state: CompositionState) -> str` — deterministic mapping from
CompositionState to ELSPETH pipeline YAML. Uses `yaml.dump()` with sorted keys
for determinism. Tests round-trip through `load_settings()`.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(web/composer): add deterministic YAML generator"
```

### Task 4.4: ComposerService — LLM Tool-Use Loop

**Files:**
- Create: `src/elspeth/web/composer/protocol.py`
- Create: `src/elspeth/web/composer/prompts.py`
- Create: `src/elspeth/web/composer/service.py`
- Create: `tests/unit/web/composer/test_service.py`

- [ ] **Step 1: Write composer loop tests**

Use a mock LLM that returns scripted tool calls. Test:
- Single tool call → state updated → assistant response returned
- Multi-turn tool calls → state accumulates correctly
- Validation error in tool call → error returned to LLM → LLM self-corrects
- Max turns exceeded → `ComposerConvergenceError`
- No tool calls (just text response) → returned immediately

- [ ] **Step 2: Implement system prompt**

`build_system_prompt()` and `build_context_message(state, catalog_summary)`.

- [ ] **Step 3: Implement ComposerServiceImpl**

The bounded tool-use loop as specified in the design. Uses LiteLLM for
provider abstraction. Injects composition state as context on each turn.
Tool calls are executed against `CompositionState` + `CatalogService`.

- [ ] **Step 4: Wire message sending into session routes**

`POST /api/sessions/{id}/messages` calls `ComposerService.compose()`,
persists the assistant message and new composition state version.

- [ ] **Step 5: Run all composer tests, commit**

```bash
git commit -m "feat(web/composer): implement LLM tool-use loop"
```

---

## Phase 5: Execution

### Task 5.1: Dry-Run Validation

**Files:**
- Create: `src/elspeth/web/execution/__init__.py`
- Create: `src/elspeth/web/execution/protocol.py`
- Create: `src/elspeth/web/execution/validation.py`
- Create: `src/elspeth/web/execution/schemas.py`
- Create: `tests/unit/web/execution/test_validation.py`

- [ ] **Step 1: Write validation tests**

Test: valid CompositionState → generates YAML → builds ExecutionGraph →
validates → returns `ValidationResult(is_valid=True, checks=[...])`.

Test: invalid state (schema mismatch) → returns `ValidationResult(is_valid=False,
errors=[{node: "gate_1", message: "field 'x' not in upstream schema"}])`.

Test: plugin instantiation failure → returns error with plugin name and detail.

- [ ] **Step 2: Implement dry-run validation**

`validate_pipeline(state: CompositionState) -> ValidationResult`:

1. Generate YAML via `yaml_generator.generate_yaml(state)`
2. Parse via `load_settings()`
3. Instantiate plugins via `instantiate_plugins_from_config()`
4. Build graph via `ExecutionGraph.from_plugin_instances()`
5. Run `graph.validate()`
6. Catch and translate exceptions into `ValidationResult` with per-component attribution

This calls the real engine validation code — no parallel validation logic.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(web/execution): add dry-run validation using real engine"
```

### Task 5.2: ExecutionService — Background Execution

**Files:**
- Create: `src/elspeth/web/execution/service.py`
- Create: `src/elspeth/web/execution/progress.py`
- Create: `tests/unit/web/execution/test_service.py`

- [ ] **Step 1: Write execution tests**

Test: `execute()` starts background thread, returns run_id immediately.
Test: run status transitions through `pending → running → completed`.
Test: `cancel()` sets shutdown event, run transitions to `cancelled`.
Test: `cancel()` on a pending run (not yet started) → directly set `cancelled`.
Test: execution failure → run status `failed` with error message.
Test: unhandled exception in `_run_pipeline()` → `done_callback` fires,
  run status is `failed` (not stuck in `running`).
Test: second `execute()` on same session while run is active → `RunAlreadyActiveError`.
Test: `ProgressBroadcaster.broadcast()` from a background thread delivers
  events to subscribed asyncio queues via `call_soon_threadsafe()`.

- [ ] **Step 2: Implement ProgressBroadcaster (B1 fix — thread-safe)**

In-process event broadcaster with explicit async/thread bridge.

Key design: the broadcaster holds a dict of `run_id → set[asyncio.Queue]`
and a reference to the asyncio event loop (captured at construction time on
the main thread).

```python
class ProgressBroadcaster:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        if run_id in self._subscribers:
            self._subscribers[run_id].discard(queue)

    def broadcast(self, run_id: str, event: RunEvent) -> None:
        """Thread-safe broadcast — callable from background threads."""
        for queue in self._subscribers.get(run_id, set()):
            self._loop.call_soon_threadsafe(queue.put_nowait, event)
```

**Critical:** `broadcast()` uses `self._loop.call_soon_threadsafe()` to safely
push events from the background thread into the asyncio event loop. Direct
`queue.put_nowait()` from a non-asyncio thread would corrupt the event loop.

Construct the broadcaster in the app factory with `asyncio.get_event_loop()`.

- [ ] **Step 3: Implement ExecutionServiceImpl (B2 + B3 + B7 fixes)**

`execute(session_id, state_id) -> run_id`:

1. Check for active runs on this session (B6 enforcement) — raise
   `RunAlreadyActiveError` if one exists
2. Load composition state from SessionService
3. Generate YAML
4. Create Run record (status=pending)
5. Create `threading.Event` for shutdown signalling
6. Store shutdown event: `self._shutdown_events[run_id] = shutdown_event`
7. Submit to thread pool with `done_callback`:
   ```python
   future = self._executor.submit(
       self._run_pipeline, run_id, yaml, shutdown_event
   )
   future.add_done_callback(self._on_pipeline_done)
   ```
8. Return run_id

**B7 fix — `_on_pipeline_done()` callback:**
```python
def _on_pipeline_done(self, future: Future) -> None:
    """Catch exceptions from background thread that bypassed try/finally."""
    exc = future.exception()
    if exc is not None:
        slog.error("Pipeline thread raised unhandled exception", exc_info=exc)
        # The try/finally in _run_pipeline should have handled this,
        # but this is the safety net.
```

**`_run_pipeline()` runs in background thread (B2 + B3 + B7 fixes):**
```python
def _run_pipeline(
    self, run_id: str, pipeline_yaml: str, shutdown_event: threading.Event
) -> None:
    try:
        self._update_run_status(run_id, "running")

        # B3 fix — construct LandscapeDB and PayloadStore from WebSettings
        landscape_db = LandscapeDB(url=self._settings.get_landscape_url())
        payload_store = FilesystemPayloadStore(
            base_path=self._settings.get_payload_store_path()
        )

        # Build graph + instantiate plugins
        settings = load_settings(yaml_content=pipeline_yaml)
        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(...)

        # B2 fix — ALWAYS pass shutdown_event to suppress signal handler
        # installation from background thread (Python forbids signal.signal()
        # from non-main threads)
        orchestrator = Orchestrator(db=landscape_db, ...)
        result = orchestrator.run(
            config=...,
            graph=graph,
            payload_store=payload_store,
            shutdown_event=shutdown_event,  # B2: suppresses SIGTERM handler
            progress_callback=lambda event: self._broadcaster.broadcast(
                run_id, self._to_run_event(event)
            ),
        )

        self._update_run_status(run_id, "completed", result=result)
    except BaseException as exc:
        # B7 fix — unconditionally write terminal status on ANY exception,
        # including KeyboardInterrupt, SystemExit, OOM-triggered exceptions.
        self._update_run_status(run_id, "failed", error=str(exc))
        raise  # Re-raise so future.add_done_callback sees it
    finally:
        self._shutdown_events.pop(run_id, None)
```

**`cancel(run_id)` implementation:**
```python
def cancel(self, run_id: str) -> None:
    event = self._shutdown_events.get(run_id)
    if event is not None:
        event.set()
    else:
        # Run may be pending (not yet started) — update directly
        self._update_run_status(run_id, "cancelled")
```

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat(web/execution): add background execution with progress"
```

### Task 5.3: Execution Routes and WebSocket

**Files:**
- Create: `src/elspeth/web/execution/routes.py`
- Create: `tests/unit/web/execution/test_routes.py`
- Create: `tests/unit/web/execution/test_progress.py`

- [ ] **Step 1: Write route tests**

Test validate endpoint, execute endpoint (returns run_id), run status endpoint,
cancel endpoint. WebSocket test: connect, receive progress events, disconnect.

- [ ] **Step 2: Implement routes**

`POST /api/sessions/{id}/validate` — calls `validation.validate_pipeline()`.
`POST /api/sessions/{id}/execute` — calls `ExecutionService.execute()`.
`GET /api/runs/{id}` — returns `RunStatus`.
`POST /api/runs/{id}/cancel` — calls `ExecutionService.cancel()`.
`GET /api/runs/{id}/results` — returns final run summary.
`WS /ws/runs/{id}` — subscribes to ProgressBroadcaster, streams RunEvents.

- [ ] **Step 3: Register in app factory, run all execution tests, commit**

```bash
git commit -m "feat(web/execution): add execution routes and WebSocket"
```

### Task 5.4: Integration Test — End-to-End Pipeline

**Files:**
- Create: `tests/integration/web/test_execute_pipeline.py`

- [ ] **Step 1: Write end-to-end test**

Test the full flow with a real (simple) pipeline — CSV source → passthrough
transform → CSV sink:

1. Create session
2. Save a composition state (manually, not via composer — tests the execution
   path independently)
3. Validate → passes
4. Execute → get run_id
5. Poll status → eventually `completed`
6. Check run results: rows_processed > 0, rows_failed == 0
7. Verify landscape_run_id links to a real audit trail

Uses a test CSV file and temp directory for sink output.

- [ ] **Step 2: Run integration test, commit**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v
git commit -m "test(web): add end-to-end pipeline execution integration test"
```

---

## Phase 6: Frontend

### Task 6.1: Project Setup

**Files:**
- Create: `src/elspeth/web/frontend/package.json`
- Create: `src/elspeth/web/frontend/tsconfig.json`
- Create: `src/elspeth/web/frontend/vite.config.ts`
- Create: `src/elspeth/web/frontend/index.html`
- Create: `src/elspeth/web/frontend/src/main.tsx`
- Create: `src/elspeth/web/frontend/src/App.tsx`

- [ ] **Step 1: Initialize Vite project**

```bash
cd src/elspeth/web/frontend
npm create vite@latest . -- --template react-ts
```

- [ ] **Step 2: Install dependencies**

```bash
npm install @xyflow/react zustand
npm install -D @types/react @types/react-dom
```

- [ ] **Step 3: Configure Vite proxy**

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
```

- [ ] **Step 4: Create minimal App with three-panel layout**

Skeleton `App.tsx` with sidebar, chat panel, and inspector panel placeholders.
Verify it renders in browser.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(web/frontend): initialize React/TypeScript project with Vite"
```

### Task 6.2: TypeScript Types and API Client

**Files:**
- Create: `src/elspeth/web/frontend/src/types/index.ts`
- Create: `src/elspeth/web/frontend/src/api/client.ts`
- Create: `src/elspeth/web/frontend/src/api/websocket.ts`

- [ ] **Step 1: Define TypeScript types**

Types matching the backend Pydantic schemas: `Session`, `ChatMessage`,
`CompositionState`, `NodeSpec`, `EdgeSpec`, `Run`, `RunEvent`,
`ValidationResult`, `PluginSummary`, `UserProfile`.

- [ ] **Step 2: Implement API client**

Typed `fetch` wrappers for all REST endpoints. Handles auth token injection,
error responses. Export functions like `createSession()`, `sendMessage()`,
`validatePipeline()`, `executePipeline()`, etc.

- [ ] **Step 3: Implement WebSocket manager**

`connectToRun(runId)` → returns event stream. Auto-reconnect on disconnect.
Parses `RunEvent` JSON. Exposes `onProgress`, `onError`, `onComplete` callbacks.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(web/frontend): add typed API client and WebSocket manager"
```

### Task 6.3: Zustand Stores

**Files:**
- Create: `src/elspeth/web/frontend/src/stores/authStore.ts`
- Create: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Create: `src/elspeth/web/frontend/src/stores/executionStore.ts`

- [ ] **Step 1: Auth store**

State: `token`, `user`, `isAuthenticated`. Actions: `login()`, `logout()`,
`loadFromStorage()`. Persists token to localStorage.

- [ ] **Step 2: Session store**

State: `sessions`, `activeSessionId`, `messages`, `compositionState`.
Actions: `createSession()`, `selectSession()`, `sendMessage()` (calls API,
updates messages + state), `loadSessions()`.

- [ ] **Step 3: Execution store**

State: `runs`, `activeRunId`, `progress`, `validationResult`.
Actions: `validate()`, `execute()`, `connectWebSocket()`, `cancel()`.
WebSocket events update `progress` reactively.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(web/frontend): add Zustand stores for auth, sessions, execution"
```

### Task 6.4: Auth Components

**Files:**
- Create: `src/elspeth/web/frontend/src/components/common/AuthGuard.tsx`
- Create: `src/elspeth/web/frontend/src/components/auth/LoginPage.tsx`
- Create: `src/elspeth/web/frontend/src/hooks/useAuth.ts`

- [ ] **Step 1: Implement AuthGuard**

Wrapper component. If not authenticated, shows LoginPage.
If authenticated, renders children.

- [ ] **Step 2: Implement LoginPage**

Simple username/password form for local auth. Calls `authStore.login()`.
For OIDC/Entra, shows "Sign in with SSO" button that redirects to IdP.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(web/frontend): add auth guard and login page"
```

### Task 6.5: Session Sidebar

**Files:**
- Create: `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`
- Create: `src/elspeth/web/frontend/src/hooks/useSession.ts`

- [ ] **Step 1: Implement SessionSidebar**

Lists user's sessions. Active session highlighted. "New session" button at
bottom. Click to switch active session. Loads sessions on mount.

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(web/frontend): add session sidebar"
```

### Task 6.6: Chat Panel

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/ComposingIndicator.tsx`
- Create: `src/elspeth/web/frontend/src/hooks/useComposer.ts`

- [ ] **Step 1: Implement MessageBubble**

Renders a single chat message. User messages right-aligned, assistant messages
left-aligned. Tool call results shown as collapsible sections.

- [ ] **Step 2: Implement ComposingIndicator**

Typing dots animation. Three dots with staggered CSS animation.
Shown when `sessionStore.isComposing` is true.

- [ ] **Step 3: Implement ChatInput**

Text input + send button. Send button disabled while composing.
Enter to send, Shift+Enter for newline.

- [ ] **Step 4: Implement ChatPanel**

Combines MessageBubble list, ComposingIndicator, and ChatInput.
Auto-scrolls to bottom on new messages. Loads message history
when active session changes.

- [ ] **Step 5: Implement useComposer hook**

Wraps `sessionStore.sendMessage()`. Sets `isComposing` while the API
call is in flight. On response, updates messages and composition state.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(web/frontend): add chat panel with composing indicator"
```

### Task 6.7: Inspector Panel — Spec View with Component Linking

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`
- Create: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx`

- [ ] **Step 1: Implement InspectorPanel**

Tab container with four tabs: Spec, Graph, YAML, Runs. Renders the active
tab's content. Shows validation status badge in header.

- [ ] **Step 2: Implement SpecView with component linking**

Renders `CompositionState` as a list of component cards:
- Each card shows type badge (colour-coded), plugin name, config summary
- Next-node indicators: `↓ next_node`
- Gate route indicators: `✓ true → sink_a`, `✗ false → sink_b`
- Error sink indicators: `⚠ on_error → error_sink`

**Click-to-highlight:**
- Track `selectedNodeId` in local state
- On click: compute upstream/downstream from edges in CompositionState
- Selected node: highlighted border + SELECTED badge
- Upstream: INPUT badge
- Downstream: OUTPUT / route-path badges
- Unrelated: dim to 35% opacity
- Click again or background: deselect

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(web/frontend): add inspector with spec view and component linking"
```

### Task 6.8: Inspector Panel — Graph, YAML, Runs Tabs

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`
- Create: `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx`
- Create: `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`

- [ ] **Step 1: Implement GraphView**

React Flow integration. Converts `CompositionState` nodes/edges to React Flow
format. Colour-coded node types. Read-only (no drag-to-connect). Pan and zoom.
Auto-layout using dagre or elkjs.

- [ ] **Step 2: Implement YamlView**

Read-only code display of generated YAML. Copy-to-clipboard button.
Syntax highlighting with a lightweight library (e.g., prism-react-renderer).

- [ ] **Step 3: Implement RunsView**

List of runs for the current session. Shows status, row counts, duration.
Click a run to show the progress detail view.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(web/frontend): add graph, YAML, and runs inspector tabs"
```

### Task 6.9: Execution UX — Validation and Progress

**Files:**
- Create: `src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx`
- Create: `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx`

- [ ] **Step 1: Implement ValidationResult**

Green pass banner or red fail banner with per-component errors and suggested
fixes. Shown inline in the inspector when validation completes.

- [ ] **Step 2: Implement ProgressView**

Live execution view: status indicator, progress bar, row count / failure
count / estimated total, recent exceptions list, cancel button.
Connects to WebSocket via `executionStore.connectWebSocket()`.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(web/frontend): add validation result and execution progress views"
```

### Task 6.10: Three-Panel Layout and Static File Serving

**Files:**
- Create: `src/elspeth/web/frontend/src/components/common/Layout.tsx`
- Modify: `src/elspeth/web/frontend/src/App.tsx`
- Modify: `src/elspeth/web/app.py`

- [ ] **Step 1: Implement Layout component**

Three-panel CSS grid: sidebar (200px fixed), chat (flex), inspector (320px fixed).
Responsive: on narrow screens, inspector collapses to a slide-over.

- [ ] **Step 2: Wire App.tsx**

Compose: `AuthGuard` → `Layout` → `SessionSidebar` + `ChatPanel` + `InspectorPanel`.
Route handling if needed (likely single-page, no router in v1).

- [ ] **Step 3: Build frontend**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 4: Add static file serving to FastAPI**

In `app.py`, mount `frontend/dist/` as static files. Serve `index.html`
for any non-API route (SPA fallback).

```python
from fastapi.staticfiles import StaticFiles

# After all API routes are registered:
frontend_dir = Path(__file__).parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True))
```

- [ ] **Step 5: Run full application, verify end-to-end**

```bash
elspeth web --port 8000
# Open http://localhost:8000
# Login, create session, send message, see spec update
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(web): complete three-panel layout with static file serving"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All spec sections mapped to tasks — pre-work (0.1),
  auth (2.1-2.4), sessions (2.5-2.6), catalog (3.1-3.2), composer (4.1-4.4),
  execution (5.1-5.4), frontend (6.1-6.10), foundation (1.1-1.3). File upload
  covered in 2.6. Composing indicator in 6.6. Component linking in 6.7.
  WebSocket progress in 5.2/6.9.
- [x] **Placeholder scan:** No TBD/TODO. Code blocks in implementation steps show patterns
  not placeholders. Some steps describe implementation intent rather than full code —
  acceptable for a plan of this size; individual task execution will fill in details.
- [x] **Type consistency:** `CompositionState` used consistently across composer and execution.
  `ToolResult` defined in 4.2, used in 4.4. `RunEvent` defined in 5.1 schemas, used in
  5.2 broadcaster and 6.9 frontend. `AuthProvider` protocol defined in 2.1, implementations
  in 2.2-2.3, middleware in 2.4.
- [x] **Name: `[webui]` extra** used consistently (not `[web]` which is taken by web scraping pack).
- [x] **Blocking review fixes applied:**
  - B1: `ProgressBroadcaster` uses `loop.call_soon_threadsafe()` (Task 5.2)
  - B2: `shutdown_event` always passed to `orchestrator.run()` (Task 5.2)
  - B3: `landscape_url` and `payload_store_path` in `WebSettings` (Task 1.1)
  - B4: Plugin manager singleton extracted to `manager.py` (Task 0.1)
  - B5: Upload path sanitized with `Path(user_id).name` + traversal test (Task 2.6)
  - B6: One-active-run enforcement via check-and-set + partial index (Task 2.5)
  - B7: `try/BaseException/finally` + `future.add_done_callback()` (Task 5.2)
  - B8: `@pytest.mark.asyncio` on all async tests + global note
