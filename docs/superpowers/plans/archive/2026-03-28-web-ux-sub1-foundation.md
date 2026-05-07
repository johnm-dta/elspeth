# Web UX Sub-Plan 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the plugin manager singleton from `cli.py` so both CLI and web entry points can share it, then create the `src/elspeth/web/` package with config, app factory, health endpoint, `[webui]` packaging extra, and `elspeth web` CLI subcommand.

**Architecture:** Phase 0 moves the `_get_plugin_manager()` singleton from `cli.py` (L3 CLI) to `plugins/infrastructure/manager.py` (L3 plugins) so both CLI and web (L3 peers) can import it without cross-dependency. Phase 1 creates the `web/` package at L3 with a Pydantic config model, a FastAPI app factory, and a Typer CLI entry point that launches uvicorn.

**Tech Stack:** FastAPI, uvicorn, Pydantic v2, Typer, pluggy (existing)

**Spec:** `docs/superpowers/specs/2026-03-28-web-ux-sub1-foundation-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/elspeth/plugins/infrastructure/manager.py` | Add `get_shared_plugin_manager()` singleton at module level |
| Modify | `src/elspeth/cli_helpers.py` | Switch import from `cli._get_plugin_manager` to `manager.get_shared_plugin_manager` |
| Modify | `src/elspeth/cli.py` | Delete `_get_plugin_manager()` / `_plugin_manager_cache`, use `get_shared_plugin_manager()` at call sites, add `web` command |
| Modify | `pyproject.toml` | Add `webui` optional extra, update `all` extra |
| Create | `src/elspeth/web/__init__.py` | Package init |
| Create | `src/elspeth/web/config.py` | `WebSettings` Pydantic model with derived accessors |
| Create | `src/elspeth/web/app.py` | `create_app()` FastAPI application factory |
| Create | `src/elspeth/web/dependencies.py` | `get_settings()` FastAPI dependency provider |
| Create | `tests/unit/plugins/test_manager_singleton.py` | Tests for `get_shared_plugin_manager()` |
| Create | `tests/unit/web/__init__.py` | Test package init |
| Create | `tests/unit/web/test_config.py` | Tests for `WebSettings` |
| Create | `tests/unit/web/test_app.py` | Tests for app factory and health endpoint |

---

## Task 0.1: Extract Plugin Manager Singleton

**Why:** `instantiate_plugins_from_config()` in `cli_helpers.py` imports `_get_plugin_manager` from `cli.py`. The web module is an L3 peer of the CLI module -- neither may import from the other. The singleton must live in `plugins/infrastructure/manager.py` where `PluginManager` is already defined.

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/manager.py`
- Modify: `src/elspeth/cli_helpers.py`
- Modify: `src/elspeth/cli.py`
- Create: `tests/unit/plugins/test_manager_singleton.py`

- [ ] **Step 1: Write the singleton test**

```python
# tests/unit/plugins/test_manager_singleton.py
"""Tests for the shared plugin manager singleton."""

from __future__ import annotations

from elspeth.plugins.infrastructure.manager import PluginManager, get_shared_plugin_manager


class TestGetSharedPluginManager:
    """Tests for get_shared_plugin_manager()."""

    def test_returns_plugin_manager_instance(self) -> None:
        pm = get_shared_plugin_manager()
        assert isinstance(pm, PluginManager)

    def test_returns_same_instance_on_repeated_calls(self) -> None:
        pm1 = get_shared_plugin_manager()
        pm2 = get_shared_plugin_manager()
        assert pm1 is pm2

    def test_has_builtin_sources_registered(self) -> None:
        pm = get_shared_plugin_manager()
        assert len(pm.get_sources()) > 0

    def test_has_builtin_transforms_registered(self) -> None:
        pm = get_shared_plugin_manager()
        assert len(pm.get_transforms()) > 0

    def test_has_builtin_sinks_registered(self) -> None:
        pm = get_shared_plugin_manager()
        assert len(pm.get_sinks()) > 0
```

- [ ] **Step 2: Run test -- verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/plugins/test_manager_singleton.py -v
```

Expected: `ImportError: cannot import name 'get_shared_plugin_manager'`

- [ ] **Step 3: Add `get_shared_plugin_manager()` to `manager.py`**

Append to the end of `src/elspeth/plugins/infrastructure/manager.py` (after the `PluginManager` class):

```python
# --- Shared singleton ---

_shared_instance: PluginManager | None = None


def get_shared_plugin_manager() -> PluginManager:
    """Return the shared plugin manager singleton.

    Creates a PluginManager and calls register_builtin_plugins() on first
    invocation.  Returns the same instance on all subsequent calls.
    Used by both CLI and web entry points.
    """
    global _shared_instance  # noqa: PLW0603
    if _shared_instance is None:
        _shared_instance = PluginManager()
        _shared_instance.register_builtin_plugins()
    return _shared_instance
```

- [ ] **Step 4: Run test -- verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/plugins/test_manager_singleton.py -v
```

Expected: All 5 tests pass.

- [ ] **Step 5: Update `cli_helpers.py` to use the new function**

In `src/elspeth/cli_helpers.py`, find and replace the import inside `instantiate_plugins_from_config()`:

Old code (around line 64):
```python
    from elspeth.cli import _get_plugin_manager
    from elspeth.core.dag import WiredTransform

    manager = _get_plugin_manager()
```

New code:
```python
    from elspeth.core.dag import WiredTransform
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    manager = get_shared_plugin_manager()
```

- [ ] **Step 6: Remove `_get_plugin_manager()` and `_plugin_manager_cache` from `cli.py`, update call sites**

In `src/elspeth/cli.py`:

1. **Delete** the module-level cache variable and the function (lines 50-68):

```python
# DELETE these lines entirely:
# Module-level singleton for plugin manager
_plugin_manager_cache: PluginManager | None = None


def _get_plugin_manager() -> PluginManager:
    """Get initialized plugin manager (singleton).

    Returns:
        PluginManager with all built-in plugins registered
    """
    global _plugin_manager_cache

    from elspeth.plugins.infrastructure.manager import PluginManager

    if _plugin_manager_cache is None:
        manager = PluginManager()
        manager.register_builtin_plugins()
        _plugin_manager_cache = manager
    return _plugin_manager_cache
```

2. **Update call site at line 1257** (in `_get_all_plugin_info()`):

Old:
```python
    manager = _get_plugin_manager()
```

New:
```python
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    manager = get_shared_plugin_manager()
```

3. **Update call site at line 2157** (in `doctor` command):

Old:
```python
        manager = _get_plugin_manager()
```

New:
```python
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        manager = get_shared_plugin_manager()
```

4. **Remove `PluginManager` from the `TYPE_CHECKING` imports** if it is no longer referenced anywhere in `cli.py` after these changes. Check whether `PluginManager` appears in any remaining type annotations in the file. If not, delete the line:

```python
    from elspeth.plugins.infrastructure.manager import PluginManager
```

from the `if TYPE_CHECKING:` block.

- [ ] **Step 7: Run full test suite -- verify no regressions**

```bash
.venv/bin/python -m pytest tests/ -x --timeout=120
```

Expected: All tests pass. The extraction is a pure refactor -- same behaviour, different location.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/plugins/infrastructure/manager.py src/elspeth/cli_helpers.py src/elspeth/cli.py tests/unit/plugins/test_manager_singleton.py
git commit -m "refactor: extract plugin manager singleton from cli.py to manager.py (B4)"
```

---

## Task 1.1: Package Structure and WebSettings Config

**Why:** The `web/` package needs a Pydantic config model that holds all settings for the web application. This is the foundation every subsequent phase depends on.

**Files:**
- Create: `src/elspeth/web/__init__.py`
- Create: `src/elspeth/web/config.py`
- Create: `tests/unit/web/__init__.py`
- Create: `tests/unit/web/test_config.py`

- [ ] **Step 1: Write config tests**

```python
# tests/unit/web/test_config.py
"""Tests for WebSettings configuration model."""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.web.config import WebSettings


class TestWebSettingsDefaults:
    """Tests for default field values."""

    def test_host_default(self) -> None:
        settings = WebSettings()
        assert settings.host == "127.0.0.1"

    def test_port_default(self) -> None:
        settings = WebSettings()
        assert settings.port == 8000

    def test_auth_provider_default(self) -> None:
        settings = WebSettings()
        assert settings.auth_provider == "local"

    def test_cors_origins_default(self) -> None:
        settings = WebSettings()
        assert settings.cors_origins == ["http://localhost:5173"]

    def test_data_dir_default(self) -> None:
        settings = WebSettings()
        assert settings.data_dir == Path("data")

    def test_composer_model_default(self) -> None:
        settings = WebSettings()
        assert settings.composer_model == "gpt-4o"

    def test_composer_max_turns_default(self) -> None:
        settings = WebSettings()
        assert settings.composer_max_turns == 20

    def test_composer_timeout_seconds_default(self) -> None:
        settings = WebSettings()
        assert settings.composer_timeout_seconds == 120.0

    def test_composer_rate_limit_per_minute_default(self) -> None:
        settings = WebSettings()
        assert settings.composer_rate_limit_per_minute == 10

    def test_secret_key_default(self) -> None:
        settings = WebSettings()
        assert settings.secret_key == "change-me-in-production"

    def test_max_upload_bytes_default(self) -> None:
        settings = WebSettings()
        assert settings.max_upload_bytes == 104857600  # 100 MB

    def test_landscape_url_default_is_none(self) -> None:
        settings = WebSettings()
        assert settings.landscape_url is None

    def test_payload_store_path_default_is_none(self) -> None:
        settings = WebSettings()
        assert settings.payload_store_path is None

    def test_oidc_fields_default_none(self) -> None:
        settings = WebSettings()
        assert settings.oidc_issuer is None
        assert settings.oidc_audience is None
        assert settings.oidc_client_id is None

    def test_entra_tenant_id_default_none(self) -> None:
        settings = WebSettings()
        assert settings.entra_tenant_id is None

    def test_session_db_url_default_is_none(self) -> None:
        settings = WebSettings()
        assert settings.session_db_url is None


class TestWebSettingsCustomValues:
    """Tests for custom field overrides."""

    def test_custom_port_and_host(self) -> None:
        settings = WebSettings(port=9090, host="0.0.0.0")
        assert settings.port == 9090
        assert settings.host == "0.0.0.0"

    def test_auth_provider_oidc(self) -> None:
        settings = WebSettings(auth_provider="oidc")
        assert settings.auth_provider == "oidc"

    def test_auth_provider_entra(self) -> None:
        settings = WebSettings(auth_provider="entra")
        assert settings.auth_provider == "entra"

    def test_custom_cors_origins(self) -> None:
        settings = WebSettings(cors_origins=["https://app.example.com", "https://staging.example.com"])
        assert len(settings.cors_origins) == 2
        assert "https://app.example.com" in settings.cors_origins

    def test_explicit_landscape_url(self) -> None:
        settings = WebSettings(landscape_url="postgresql://db/audit")
        assert settings.landscape_url == "postgresql://db/audit"

    def test_explicit_payload_store_path(self) -> None:
        settings = WebSettings(payload_store_path=Path("/mnt/payloads"))
        assert settings.payload_store_path == Path("/mnt/payloads")


class TestWebSettingsValidation:
    """Tests for field validation."""

    def test_invalid_auth_provider_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(auth_provider="invalid")  # type: ignore[arg-type]

    def test_invalid_auth_provider_kerberos_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(auth_provider="kerberos")  # type: ignore[arg-type]


class TestWebSettingsDerivedAccessors:
    """Tests for get_landscape_url() and get_payload_store_path()."""

    def test_get_landscape_url_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(data_dir=Path("/app/data"))
        url = settings.get_landscape_url()
        assert url == "sqlite:////app/data/runs/audit.db"

    def test_get_landscape_url_explicit_value_returned(self) -> None:
        settings = WebSettings(landscape_url="postgresql://db/audit")
        url = settings.get_landscape_url()
        assert url == "postgresql://db/audit"

    def test_get_payload_store_path_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(data_dir=Path("/app/data"))
        path = settings.get_payload_store_path()
        assert path == Path("/app/data/payloads")

    def test_get_payload_store_path_explicit_value_returned(self) -> None:
        settings = WebSettings(payload_store_path=Path("/mnt/payloads"))
        path = settings.get_payload_store_path()
        assert path == Path("/mnt/payloads")

    def test_default_data_dir_landscape_url(self) -> None:
        """Default data_dir='data' produces a relative sqlite path."""
        settings = WebSettings()
        url = settings.get_landscape_url()
        assert url == f"sqlite:///{Path('data') / 'runs' / 'audit.db'}"

    def test_default_data_dir_payload_store_path(self) -> None:
        """Default data_dir='data' produces a relative payload path."""
        settings = WebSettings()
        path = settings.get_payload_store_path()
        assert path == Path("data") / "payloads"

    def test_get_session_db_url_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(data_dir=Path("/app/data"))
        url = settings.get_session_db_url()
        assert url == "sqlite:////app/data/sessions.db"

    def test_get_session_db_url_explicit_value_returned(self) -> None:
        settings = WebSettings(session_db_url="postgresql://db/sessions")
        url = settings.get_session_db_url()
        assert url == "postgresql://db/sessions"

    def test_default_data_dir_session_db_url(self) -> None:
        """Default data_dir='data' produces a relative sqlite path."""
        settings = WebSettings()
        url = settings.get_session_db_url()
        assert url == f"sqlite:///{Path('data') / 'sessions.db'}"
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web'`

- [ ] **Step 3: Create the web package and WebSettings**

```python
# src/elspeth/web/__init__.py
"""ELSPETH Web UX -- LLM Composer MVP."""
```

```python
# tests/unit/web/__init__.py
```

```python
# src/elspeth/web/config.py
"""Web application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class WebSettings(BaseModel):
    """Configuration for the ELSPETH web application.

    All fields have sensible defaults for local development.
    auth_provider uses a Literal type so Pydantic rejects invalid
    values automatically -- no manual @field_validator needed.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    auth_provider: Literal["local", "oidc", "entra"] = "local"
    cors_origins: list[str] = ["http://localhost:5173"]
    data_dir: Path = Path("data")
    composer_model: str = "gpt-4o"
    composer_max_turns: int = 20
    composer_timeout_seconds: float = 120.0
    composer_rate_limit_per_minute: int = 10
    secret_key: str = "change-me-in-production"  # S3: startup guard in Sub-2 enforces non-default in production
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB

    # Execution infrastructure (B3 fix)
    # Defaults derive from data_dir when not explicitly set
    landscape_url: str | None = None
    payload_store_path: Path | None = None

    # OIDC / Entra-specific (optional)
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_client_id: str | None = None
    entra_tenant_id: str | None = None

    # Session database (sessions, messages, composition states, runs)
    # Separate from landscape_url (audit DB)
    session_db_url: str | None = None

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

    def get_session_db_url(self) -> str:
        """Resolve session DB URL, defaulting to data_dir-relative path."""
        if self.session_db_url is not None:
            return self.session_db_url
        db_path = self.data_dir / "sessions.db"
        return f"sqlite:///{db_path}"
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/test_config.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/__init__.py src/elspeth/web/config.py tests/unit/web/__init__.py tests/unit/web/test_config.py
git commit -m "feat(web): add web package with WebSettings config"
```

---

## Task 1.2: FastAPI Application Factory

**Why:** The app factory is the central integration point -- it wires CORS middleware, stores settings on `app.state`, and registers the health endpoint. Every subsequent phase registers its router here.

**Files:**
- Create: `src/elspeth/web/app.py`
- Create: `src/elspeth/web/dependencies.py`
- Create: `tests/unit/web/test_app.py`

- [ ] **Step 1: Write app factory tests**

```python
# tests/unit/web/test_app.py
"""Tests for the FastAPI application factory."""

from __future__ import annotations

from starlette.testclient import TestClient

from elspeth.web.app import create_app
from elspeth.web.config import WebSettings


class TestCreateApp:
    """Tests for create_app()."""

    def test_returns_fastapi_instance_with_correct_title(self) -> None:
        app = create_app(WebSettings())
        assert app.title == "ELSPETH Web"

    def test_returns_fastapi_instance_with_correct_version(self) -> None:
        app = create_app(WebSettings())
        assert app.version == "0.1.0"

    def test_default_settings_when_none_passed(self) -> None:
        app = create_app()
        assert app.state.settings.port == 8000

    def test_settings_stored_on_app_state(self) -> None:
        settings = WebSettings(port=9999)
        app = create_app(settings)
        assert app.state.settings is settings
        assert app.state.settings.port == 9999


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self) -> None:
        app = create_app(WebSettings())
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self) -> None:
        app = create_app(WebSettings())
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.json() == {"status": "ok"}


class TestCORSMiddleware:
    """Tests that CORS middleware is configured."""

    def test_cors_allows_configured_origin(self) -> None:
        settings = WebSettings(cors_origins=["http://localhost:5173"])
        app = create_app(settings)
        client = TestClient(app)
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_rejects_unconfigured_origin(self) -> None:
        settings = WebSettings(cors_origins=["http://localhost:5173"])
        app = create_app(settings)
        client = TestClient(app)
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Starlette CORS middleware omits the header for disallowed origins
        assert "access-control-allow-origin" not in response.headers
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/test_app.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.app'`

- [ ] **Step 3: Implement the app factory and dependencies module**

```python
# src/elspeth/web/app.py
"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from elspeth.web.config import WebSettings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Async lifespan context manager for the FastAPI application.

    Services that require a running event loop must be constructed here,
    not in the synchronous create_app() function.

    Phase 1: stub -- no async services yet.
    Phase 5 will construct ProgressBroadcaster here using
    asyncio.get_running_loop().
    """
    # Phase 5 will add:
    #   app.state.broadcaster = ProgressBroadcaster(asyncio.get_running_loop())
    yield
    # Shutdown: clean up async services here.


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Web application settings. Constructs defaults if None.

    Returns:
        Configured FastAPI instance with CORS middleware and health endpoint.
    """
    if settings is None:
        settings = WebSettings()

    app = FastAPI(title="ELSPETH Web", version="0.1.0", lifespan=lifespan)

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
"""FastAPI dependency injection providers.

Each phase adds its service dependency here.  Phase 1 provides only
get_settings().  Service stubs are added as phases 2-5 land.

Service construction order follows specs/2026-03-28-web-ux-seam-contracts.md.
ProgressBroadcaster and event-loop reference are created in the lifespan()
async context manager (not in the synchronous create_app()), per the seam
contract for async service wiring.
"""

from __future__ import annotations

from fastapi import Request

from elspeth.web.config import WebSettings


def get_settings(request: Request) -> WebSettings:
    """Retrieve application settings from app state.

    Intended for use as a FastAPI Depends() provider.
    """
    return request.app.state.settings
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/test_app.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
.venv/bin/python -m pytest tests/ -x --timeout=120
```

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/app.py src/elspeth/web/dependencies.py tests/unit/web/test_app.py
git commit -m "feat(web): add FastAPI app factory with health endpoint and CORS"
```

---

## Task 1.3: Packaging Extra and CLI Entry Point

**Why:** Users need `uv pip install -e ".[webui]"` to pull in FastAPI/uvicorn, and `elspeth web` to start the server. The CLI command must degrade gracefully when the extra is not installed.

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/elspeth/cli.py`

- [ ] **Step 1: Add the `webui` optional extra to `pyproject.toml`**

In `pyproject.toml`, add a new `webui` extra in the `[project.optional-dependencies]` section. Place it after the existing `security` extra and before `tracing-langfuse`:

```toml
webui = [
    # Web UX -- LLM Composer MVP
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.34,<1",
    "PyJWT[crypto]>=2.8,<3",
    "python-multipart>=0.0.20",
    "websockets>=14.0,<15",
    "httpx>=0.27,<1",
]
```

- [ ] **Step 2: Update the `all` extra to include webui dependencies**

In the `all` section of `[project.optional-dependencies]`, add webui dependencies after the existing `# web dependencies` block. Add these lines:

```toml
    # webui dependencies
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.34,<1",
    "PyJWT[crypto]>=2.8,<3",
    "python-multipart>=0.0.20",
    "websockets>=14.0,<15",
```

Note: `httpx` is already present in `all` via the llm dependencies, so it does not need to be added again.

- [ ] **Step 3: Install the new extra**

```bash
uv pip install -e ".[webui,dev]"
```

Verify:
```bash
.venv/bin/python -c "import fastapi; print(fastapi.__version__)"
.venv/bin/python -c "import uvicorn; print(uvicorn.__version__)"
```

- [ ] **Step 4: Add the `elspeth web` CLI subcommand**

In `src/elspeth/cli.py`, add the following command. Place it before the `if __name__ == "__main__":` block at the end of the file:

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
        typer.echo(
            "Error: Web UI requires the [webui] extra. "
            "Install with: uv pip install -e '.[webui]'",
            err=True,
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

- [ ] **Step 5: Verify CLI help output**

```bash
.venv/bin/python -m elspeth web --help
```

Expected output should show:
- `--port` (default 8000)
- `--host` (default 127.0.0.1)
- `--auth` (default local)
- `--reload` / `--no-reload`

- [ ] **Step 6: Smoke-test the server starts**

Start the server in the background, hit the health endpoint, then stop it:

```bash
# Start in background
.venv/bin/python -m elspeth web --port 8765 &
SERVER_PID=$!
sleep 2

# Hit health endpoint
curl -s http://127.0.0.1:8765/api/health
# Expected: {"status":"ok"}

# Clean up
kill $SERVER_PID
```

- [ ] **Step 7: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -x --timeout=120
```

- [ ] **Step 8: Run layer compliance check**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth
```

Expected: No new violations. `web/` is L3 (application layer), importing only from `web/` internal modules and Pydantic.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/elspeth/cli.py
git commit -m "feat(web): add [webui] extra and elspeth web CLI command"
```

---

## Self-Review Checklist

Before marking this sub-plan as complete, verify every item:

- [ ] **Singleton extraction:** `get_shared_plugin_manager()` exists in `manager.py` and returns the same instance on repeated calls
- [ ] **CLI decoupled:** `cli_helpers.py` no longer imports anything from `cli.py`
- [ ] **Dead code removed:** `_get_plugin_manager()` and `_plugin_manager_cache` do not exist in `cli.py`
- [ ] **WebSettings complete:** All 19 fields present (host, port, auth_provider, cors_origins, data_dir, composer_model, composer_max_turns, composer_timeout_seconds, composer_rate_limit_per_minute, secret_key, max_upload_bytes, landscape_url, payload_store_path, oidc_issuer, oidc_audience, oidc_client_id, entra_tenant_id, session_db_url)
- [ ] **No `validate_auth_provider`:** `auth_provider` uses `Literal["local", "oidc", "entra"]` -- Pydantic handles validation, no manual `@field_validator`
- [ ] **Derived accessors work:** `get_landscape_url()`, `get_payload_store_path()`, and `get_session_db_url()` return data-dir-relative defaults when fields are `None`, and return explicit values when set
- [ ] **App factory correct:** `create_app()` returns a `FastAPI` with `title="ELSPETH Web"`, `version="0.1.0"`, CORS middleware, lifespan context manager, and `/api/health` endpoint
- [ ] **Lifespan stub present:** `lifespan()` async context manager exists in `app.py`, passed to `FastAPI(lifespan=lifespan)`, with Phase 5 `ProgressBroadcaster` comments
- [ ] **No lazy imports in `create_app()`:** All imports are at the top of `app.py`, not inside the function body
- [ ] **dependencies.py exists:** `get_settings()` reads from `request.app.state.settings`
- [ ] **`[webui]` extra in pyproject.toml:** Contains fastapi, uvicorn[standard], PyJWT[crypto], python-multipart, websockets, httpx
- [ ] **`all` extra updated:** Includes webui dependencies
- [ ] **CLI command works:** `elspeth web --help` shows port, host, auth, reload options
- [ ] **Graceful degradation:** Without `[webui]` installed, `elspeth web` prints an error to stderr and exits with code 1
- [ ] **Layer compliance:** `enforce_tier_model.py` passes with no new violations
- [ ] **All tests pass:** `pytest tests/ -x` has no failures
- [ ] **No `@pytest.mark.asyncio` needed:** This sub-plan has no async test functions (all tests are synchronous)

---

## Round 4 Review Amendments

> **Status: Amendments below have been integrated into the plan body.**

Changes from the expert panel review (Round 4) that affect this sub-plan:

- **H8p — `composer_rate_limit_per_minute` field:** Task 1.1 must add `composer_rate_limit_per_minute: int = 10` to `WebSettings` in `config.py`. A corresponding default test (`test_composer_rate_limit_per_minute_default`) must be added to `tests/unit/web/test_config.py`. Update the self-review checklist field count from 18 to 19.
- **S3/C4 — `secret_key` production guard:** The `secret_key` field retains its default `"change-me-in-production"` in `WebSettings` (created in this sub-plan). The field must carry an `# S3: startup guard in Sub-2 enforces non-default in production` comment. The actual enforcement (hard `SystemExit` crash when the default value is detected in non-test environments) is implemented in Sub-2's `app.py` startup path, not here.
- **Seam contracts reference:** Task 1.2's `dependencies.py` stub must note in its module docstring that service construction order follows `specs/2026-03-28-web-ux-seam-contracts.md`. Specifically: `ProgressBroadcaster` and the event-loop reference are created in the `lifespan()` async context manager (not in the synchronous `create_app()`), per the seam contract for async service wiring.
