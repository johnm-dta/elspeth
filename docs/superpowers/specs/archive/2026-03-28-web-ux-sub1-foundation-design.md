# Web UX Sub-Spec 1: Foundation

**Status:** Draft
**Date:** 2026-03-28
**Parent Spec:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`
**Phase:** 0 + 1 (Foundation)
**Depends On:** Nothing
**Blocks:** All subsequent phases

---

## Scope

**In scope:**

- Extract the plugin manager singleton from `cli.py` so both CLI and web entry points can use it without cross-dependency (Phase 0)
- Create the `src/elspeth/web/` package with config, app factory, and health endpoint
- Add the `[webui]` optional extra to `pyproject.toml`
- Add the `elspeth web` CLI subcommand with uvicorn integration
- Health endpoint at `/api/health`
- CORS middleware configured from settings

**Out of scope:**

- Auth, sessions, catalog, composer, execution, frontend (Phases 2-6)
- Database setup or SQLAlchemy models
- Any service Protocol definitions beyond `dependencies.py` stubs
- Static file serving for the frontend SPA (added in Phase 6)

---

## Pre-work: Plugin Manager Extraction

`cli_helpers.py` currently imports `_get_plugin_manager()` from `cli.py`. The web module is an L3 peer of the CLI module -- neither may import from the other. The plugin manager singleton must live in `plugins/infrastructure/manager.py` where `PluginManager` is already defined.

**What moves:**

- The `_plugin_manager_cache` module-level variable and `_get_plugin_manager()` function move from `cli.py` to `plugins/infrastructure/manager.py` as `get_shared_plugin_manager()`.
- `cli_helpers.py` switches its import from `cli._get_plugin_manager` to `plugins.infrastructure.manager.get_shared_plugin_manager`.
- All remaining call sites in `cli.py` switch to the new function.
- `_get_plugin_manager()` and `_plugin_manager_cache` are deleted from `cli.py` entirely (no shim, no deprecation).

**Singleton contract:** `get_shared_plugin_manager()` creates a `PluginManager` on first call, calls `register_builtin_plugins()`, caches the instance at module level, and returns the same instance on subsequent calls.

---

## WebSettings

All fields on `WebSettings`, a Pydantic `BaseModel`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `host` | `str` | `"127.0.0.1"` | Bind address for uvicorn |
| `port` | `int` | `8000` | Listen port for uvicorn |
| `auth_provider` | `Literal["local", "oidc", "entra"]` | `"local"` | Which auth backend to use |
| `cors_origins` | `list[str]` | `["http://localhost:5173"]` | Allowed CORS origins (Vite dev server default) |
| `data_dir` | `Path` | `Path("data")` | Root for uploads, payloads, and audit DB |
| `composer_model` | `str` | `"gpt-4o"` | LiteLLM model identifier for the composition LLM |
| `composer_max_turns` | `int` | `20` | Maximum tool-use loop iterations before convergence error |
| `composer_timeout_seconds` | `float` | `120.0` | HTTP timeout for the composer loop (W12 fix) |
| `composer_rate_limit_per_minute` | `int` | `10` | Maximum `POST /messages` requests per user per minute. Prevents LLM cost amplification via rapid message submission (H8 fix). Enforced by the messages route handler using a per-user in-memory counter with a sliding window. Returns HTTP 429 Too Many Requests when exceeded. |
| `secret_key` | `str` | `"change-me-in-production"` | JWT signing key for local auth. **S3: App crashes on startup if this default is not changed in non-test environments.** |
| `max_upload_bytes` | `int` | `104857600` (100 MB) | Maximum file upload size (W8 fix) |
| `landscape_url` | `str \| None` | `None` | SQLAlchemy URL for the Landscape audit DB. When `None`, resolves to `sqlite:///{data_dir}/runs/audit.db` via `get_landscape_url()` (B3 fix) |
| `payload_store_path` | `Path \| None` | `None` | Directory for payload blob storage. When `None`, resolves to `{data_dir}/payloads/` via `get_payload_store_path()` (B3 fix) |
| `oidc_issuer` | `str \| None` | `None` | OIDC issuer URL (required when `auth_provider="oidc"`) |
| `oidc_audience` | `str \| None` | `None` | OIDC audience claim (required when `auth_provider="oidc"`) |
| `oidc_client_id` | `str \| None` | `None` | OIDC/Entra client ID (used by auth config endpoint and frontend OIDC redirect construction). Note: in OIDC, the audience claim is typically the client ID. `oidc_audience` is used for token validation; `oidc_client_id` is used for the auth config endpoint response and frontend redirect URL construction. They may have the same value but serve different purposes. |
| `entra_tenant_id` | `str \| None` | `None` | Azure Entra tenant ID (required when `auth_provider="entra"`) |
| `session_db_url` | `str \| None` | `None` | SQLAlchemy URL for the session database (sessions, messages, composition states, runs). When `None`, resolves to `sqlite:///{data_dir}/sessions.db` via `get_session_db_url()`. Separate from `landscape_url` (audit DB) |

**Derived accessors:**

- `get_landscape_url() -> str` -- returns `landscape_url` if set, otherwise `sqlite:///{data_dir}/runs/audit.db`
- `get_payload_store_path() -> Path` -- returns `payload_store_path` if set, otherwise `{data_dir}/payloads/`
- `get_session_db_url() -> str` -- returns `session_db_url` if set, otherwise `sqlite:///{data_dir}/sessions.db`

**Validation:** `auth_provider` uses `Literal` type, so Pydantic rejects invalid values automatically. No manual `@field_validator` needed.

---

## FastAPI Application Factory

`create_app(settings: WebSettings | None = None) -> FastAPI`

Located in `src/elspeth/web/app.py`. Responsibilities:

- Accept an optional `WebSettings` instance; construct a default if `None`
- Create a `FastAPI` instance with `title="ELSPETH Web"` and `version="0.1.0"`
- Attach CORS middleware using `settings.cors_origins`
- Store `settings` on `app.state.settings` for dependency injection
- Register the `/api/health` GET endpoint returning `{"status": "ok"}`
- Return the configured `FastAPI` instance

**Lifespan context manager:** `create_app` attaches an async lifespan context manager to the `FastAPI` instance. Services that require a running event loop (e.g., `ProgressBroadcaster` in Phase 5, which needs `asyncio.get_running_loop()`) **must** be constructed inside the lifespan's `__aenter__`, not in the synchronous `create_app()` function. The lifespan populates `app.state` with these async-dependent services.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 1: stub — no async services yet.
    # Phase 5 will add:
    #   app.state.broadcaster = ProgressBroadcaster(asyncio.get_running_loop())
    yield
    # Shutdown: clean up async services here.
```

**Critical constraint:** `create_app()` is synchronous. Do NOT call `asyncio.get_event_loop()` inside it — there may not be a running loop at factory time (e.g., during test setup or uvicorn's import phase). Async-dependent services belong in the lifespan, which runs inside the ASGI server's event loop. Synchronous state (like `settings`) can still be attached to `app.state` directly in `create_app()`.

**`dependencies.py`** provides `get_settings(request: Request) -> WebSettings` as a FastAPI `Depends()` provider that reads from `request.app.state.settings`. Service dependency stubs are added as each subsequent phase lands.

**What `create_app` does NOT do in Phase 1:** register auth middleware, mount routers, create database tables, instantiate services, or serve static files.

---

## Packaging

Add a `webui` optional extra to `pyproject.toml` under `[project.optional-dependencies]`:

| Dependency | Version Constraint | Purpose |
|------------|-------------------|---------|
| `fastapi` | `>=0.115,<1` | Web framework |
| `uvicorn[standard]` | `>=0.34,<1` | ASGI server with lifespan support |
| `PyJWT[crypto]` | `>=2.8,<3` | JWT encoding/decoding for local auth |
| `python-multipart` | `>=0.0.20` | File upload parsing |
| `websockets` | `>=14.0,<15` | WebSocket protocol support |
| `httpx` | `>=0.27,<1` | Async HTTP client (OIDC JWKS discovery, test client) |

The existing `all` extra must be updated to include `"elspeth[webui]"`.

The extra is named `webui` (not `web`) because `web` is already taken by the web scraping/HTML processing extra (`html2text`, `beautifulsoup4`).

---

## CLI Entry Point

Add an `elspeth web` command to `cli.py` as a `@app.command()`.

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | `int` | `8000` | Port to listen on |
| `--host` | `str` | `"127.0.0.1"` | Host to bind to |
| `--auth` | `str` | `"local"` | Auth provider: `local`, `oidc`, `entra` |
| `--reload` | `bool` | `False` | Enable uvicorn auto-reload for development |

**Behaviour:**

1. Attempt to import `uvicorn` and `elspeth.web.config.WebSettings`. If `ImportError`, print an error message directing the user to install the `[webui]` extra and exit with code 1.
2. Construct a `WebSettings` instance from the CLI options.
3. Call `uvicorn.run()` with the `"elspeth.web.app:create_app"` factory string, `factory=True`, and the host/port/reload options.

**W7 note:** When `--reload` is enabled, uvicorn imports the factory string in a subprocess, so `WebSettings` must be constructable without CLI arguments (using defaults). Non-default settings should be forwarded via environment variables in a later phase if needed. When `--reload` is disabled, the app instance could be passed directly, but the factory string approach is used consistently for simplicity.

---

## File Map

| Action | Path |
|--------|------|
| Modify | `src/elspeth/plugins/infrastructure/manager.py` -- add `get_shared_plugin_manager()` |
| Modify | `src/elspeth/cli_helpers.py` -- switch import to `get_shared_plugin_manager` |
| Modify | `src/elspeth/cli.py` -- delete `_get_plugin_manager()`/`_plugin_manager_cache`, add `web` command |
| Modify | `pyproject.toml` -- add `webui` extra, update `all` extra |
| Create | `src/elspeth/web/__init__.py` |
| Create | `src/elspeth/web/app.py` |
| Create | `src/elspeth/web/config.py` |
| Create | `src/elspeth/web/dependencies.py` |
| Create | `tests/unit/plugins/test_manager_singleton.py` |
| Create | `tests/unit/web/__init__.py` |
| Create | `tests/unit/web/test_app.py` |
| Create | `tests/unit/web/test_config.py` |

---

## Dependencies on Existing Code

| Existing Code | How Phase 0+1 Uses It |
|---------------|----------------------|
| `src/elspeth/plugins/infrastructure/manager.py` (`PluginManager`) | Extraction target for the singleton; `get_shared_plugin_manager()` wraps `PluginManager()` + `register_builtin_plugins()` |
| `src/elspeth/cli.py` | Remove `_get_plugin_manager()`, add `web` command alongside existing commands |
| `src/elspeth/cli_helpers.py` | Update import site from `cli._get_plugin_manager` to `manager.get_shared_plugin_manager` |
| `pyproject.toml` | Add new optional extra and update `all` group |

No engine, core, or contracts code is touched. No existing tests should break -- the plugin manager extraction is a pure refactor (same behaviour, different location).

---

## Acceptance Criteria

1. **Singleton extraction works.** `get_shared_plugin_manager()` returns the same instance on repeated calls. `cli_helpers.py` no longer imports from `cli.py`. `_get_plugin_manager` and `_plugin_manager_cache` no longer exist in `cli.py`.

2. **WebSettings validates correctly.** Default construction produces expected values. Invalid `auth_provider` values are rejected. `get_landscape_url()` and `get_payload_store_path()` return data-dir-relative defaults when their fields are `None`, and return explicit values when set.

3. **App factory produces a working FastAPI app.** `create_app()` returns a `FastAPI` instance with `title="ELSPETH Web"`. `GET /api/health` returns `200` with `{"status": "ok"}`. CORS middleware is attached with the configured origins.

4. **`[webui]` extra installs cleanly.** `uv pip install -e ".[webui,dev]"` succeeds. `import fastapi` and `import uvicorn` work after installation.

5. **CLI entry point works.** `elspeth web --help` shows port, host, auth, and reload options. Running `elspeth web` starts uvicorn on `127.0.0.1:8000` and the health endpoint responds. Without `[webui]` installed, the command prints an informative error and exits with code 1.

6. **Existing tests pass.** The full test suite (`pytest tests/`) passes with no regressions from the plugin manager extraction or new imports.

7. **Layer compliance.** `enforce_tier_model.py` passes -- `web/` is L3, no upward imports introduced.
