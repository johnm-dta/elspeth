# Server Configuration Design

**Date:** 2026-04-03
**Status:** Approved (design review complete)
**Scope:** YAML-based persistent configuration for the ELSPETH web server component

## Problem

The web server (`elspeth web`) is the only persistent service in ELSPETH, yet it has the
least sophisticated configuration mechanism. All settings are env-var-only (`ELSPETH_WEB__*`),
with no config file support. This makes reproducible deployments difficult — operators must
maintain shell scripts or `.env` files to remember 20+ environment variables across restarts.

The deployment at `elspeth.foundryside.dev` (Caddy reverse proxy to `localhost:8451`) exposed
the gap: CORS origins, trusted proxies, and secret keys all need persistent configuration
that survives process restarts without relying on env var management.

## Design Decisions

### Format: YAML

Consistent with pipeline `settings.yaml` files. `pyyaml` is already a core dependency.
Supports `${VAR}` expansion via reuse of existing `_expand_env_vars()` from `core/config.py`.
TOML was considered but rejected for consistency — ELSPETH users already know YAML.

### Config file lookup: Explicit only

No convention paths (no auto-discovery of `server.yaml` in CWD). Config file is always opt-in:

1. `--config path/to/server.yaml` CLI flag
2. `ELSPETH_SERVER_CONFIG` env var
3. No file found → defaults + `ELSPETH_WEB__*` env vars only

### Precedence (high to low)

1. CLI flags (`--port`, `--host`, `--auth`)
2. `ELSPETH_WEB__*` env vars (flat namespace, backwards-compatible)
3. Config file values
4. Pydantic defaults (fail-closed where security/reliability relevant)

### Architecture: Sectioned Pydantic models (Approach B)

Nested frozen Pydantic models mirroring YAML sections. Replaces the monolithic `WebSettings`.
Follows the pattern already established by `ElspethSettings` with nested subsystem configs.

## Trust Model

The web server is an operations interface, not the audit backbone. It's T2/T3 except for
paths that handle secrets or could compromise pipeline execution.

- **Config file parsing errors** → hard failure (T2: our format, if it doesn't parse that's real)
- **Env var wrong type** → hard failure (Pydantic rejects)
- **Unknown YAML key** → hard failure (typo detection)
- **Missing optional sections** → fine, use defaults
- **Secret values** → T1-equivalent: never log, redact in startup banner and error messages
- **Pipeline YAML handed to engine** → T1-equivalent: must be fully validated first
- **Config file might be hostile** → not a concern (write access to config = server ownership)

Full trust model documented in `tier-model-deep-dive` skill under "Web Component Trust Model".

## YAML Structure

```yaml
# server.yaml

server:
  host: 127.0.0.1
  port: 8451
  workers: 1
  trusted_proxies: []
  allowed_hosts: []
  request_timeout_seconds: 300

security:
  secret_key: ${ELSPETH_SECRET_KEY}
  auth_provider: local            # local | oidc | entra
  registration_mode: open         # open | email_verified | closed
  server_secret_allowlist:
    - OPENROUTER_API_KEY
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - AZURE_API_KEY
  # OIDC/Entra fields (required when auth_provider != local)
  oidc_issuer: null
  oidc_audience: null
  oidc_client_id: null
  oidc_authorization_endpoint: null
  entra_tenant_id: null

storage:
  data_dir: data
  landscape_url: null             # defaults to data_dir/runs/audit.db
  landscape_passphrase: null
  payload_store_path: null        # defaults to data_dir/payloads
  session_db_url: null            # defaults to data_dir/sessions.db

cors:
  origins:
    - http://localhost:5173

rate_limiting:
  requests_per_minute: 120
  burst: 20
  per_user_per_minute: 60

limits:
  max_upload_bytes: 104857600           # 100MB
  max_blob_storage_per_session_bytes: 524288000  # 500MB
  max_concurrent_runs: 4
  max_runs_per_user: 2

timeouts:
  orphan_run_max_age_seconds: 3600
  session_idle_timeout_seconds: 1800    # 30 min
  graceful_shutdown_timeout_seconds: 30

composer:
  model: gpt-4o
  max_composition_turns: 15
  max_discovery_turns: 10
  timeout_seconds: 85.0
  rate_limit_per_minute: 10

logging:
  level: INFO                     # DEBUG | INFO | WARNING | ERROR
  format: text                    # text | json
  access_log: true
```

## Pydantic Model Hierarchy

```
ServerConfig (top-level, frozen)
  +-- server: ServerSection
  +-- security: SecuritySection
  +-- storage: StorageSection
  +-- cors: CorsSection
  +-- rate_limiting: RateLimitSection
  +-- limits: LimitsSection
  +-- timeouts: TimeoutsSection
  +-- composer: ComposerSection
  +-- logging: LoggingSection
```

All section models are frozen (`ConfigDict(frozen=True)`). Each has sensible defaults so
the entire config file is optional and every section within it is optional.

## Fail-Closed Defaults

Settings where the default must be safe, not convenient:

| Setting | Default | Rationale |
|---------|---------|-----------|
| `server.workers` | `1` | Safe; scaling up is an explicit choice |
| `limits.max_concurrent_runs` | `4` | Won't OOM a modest server |
| `server.graceful_shutdown_timeout_seconds` | `30` | Drain requests without hanging indefinitely |
| `server.request_timeout_seconds` | `300` (5 min) | Prevents infinite hangs |
| `rate_limiting.requests_per_minute` | `120` | Generous for humans, stops runaway scripts |
| `logging.level` | `INFO` | Enough to diagnose without drowning in noise |
| `logging.access_log` | `true` | Always want this; turning it off is the explicit choice |
| `server.trusted_proxies` | `[]` (empty) | Trust nobody by default. Must explicitly list Caddy's IP. |
| `server.allowed_hosts` | `[]` (empty) | Permissive for dev, tighten explicitly for production |

## Validation

### Hard failures (refuse to start)

| Condition | Rationale |
|-----------|-----------|
| Non-local `host` + default `secret_key` | Existing guard, promoted to `ServerConfig` cross-section validator |
| `auth_provider` is `oidc`/`entra` + missing required fields | Existing guard, moved to `SecuritySection` |
| Port, workers out of range | Pydantic constraints (`ge=1`, `le=65535`, etc.) |
| `--config` specified but file doesn't exist | Fail-fast, don't silently fall back to defaults |
| `storage.data_dir` not writable | Catch permission issues at startup, not mid-pipeline |
| Unknown top-level YAML keys | Typo detection (mirrors `_validate_unknown_yaml_keys()` from pipeline config) |

### Startup warnings (log at WARNING, continue)

| Condition | Rationale |
|-----------|-----------|
| Non-local `host` + empty `trusted_proxies` | `X-Forwarded-For` won't be trusted, IP-based rate limiting hits Caddy's IP |
| Non-local `host` + empty `allowed_hosts` | Host header injection possible behind proxy |
| `workers` > 1 + `logging.format` is `text` | Interleaved text logs from multiple workers are unreadable |
| `rate_limiting.requests_per_minute` is 0 | Rate limiting is off — operator should know |
| `cors.origins` contains `*` | Wide-open CORS, sometimes intentional but flag it |

### Startup banner

On boot, log a structured summary of the active configuration with `secret_key` and
`landscape_passphrase` redacted. Includes config source path.

```
INFO  ELSPETH web server starting
INFO  Config source: /etc/elspeth/server.yaml
INFO  server.host=0.0.0.0  server.port=8451  server.workers=2
INFO  security.auth_provider=oidc  security.secret_key=<redacted>
INFO  storage.data_dir=/var/lib/elspeth
INFO  logging.level=INFO  logging.format=json  logging.access_log=true
INFO  rate_limiting.requests_per_minute=120
INFO  limits.max_concurrent_runs=4
WARN  server.trusted_proxies is empty -- X-Forwarded-For headers will not be trusted
```

## CLI Integration

### Command signature change

```python
@app.command()
def web(
    config: Path | None = typer.Option(None, help="Server config file"),
    port: int | None = typer.Option(None, help="Port to listen on"),
    host: str | None = typer.Option(None, help="Host to bind to"),
    auth: str | None = typer.Option(None, help="Auth provider: local, oidc, entra"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
```

CLI flags become `None`-defaulted (not value-defaulted) so we can distinguish "user passed
`--port 9000`" from "user didn't pass `--port`". Actual defaults live in Pydantic models.

### New `elspeth web init` subcommand

Generates a commented `server.yaml` with all defaults and explanations. This is the primary
documentation for the config file — operators read the generated file.

```bash
$ elspeth web init
Wrote server.yaml with default configuration.
Edit this file, then start with: elspeth web --config server.yaml
```

### Env var bridge

The factory protocol (`uvicorn.run(..., factory=True)`) still needs no-argument construction.
With `--config`, the CLI sets `ELSPETH_SERVER_CONFIG` pointing to the resolved config path,
plus any CLI overrides as `ELSPETH_WEB__*` env vars. The factory calls `load_server_config()`
which reads all three sources.

### Flat env var mapping

`ELSPETH_WEB__PORT=8451` maps to `server.port`. A hardcoded mapping dict translates flat
env var names to sectioned paths. This preserves backwards compatibility without inventing
fragile `__` nesting conventions.

```python
ENV_VAR_MAPPING: dict[str, tuple[str, str]] = {
    # env var suffix -> (section, field)
    "HOST": ("server", "host"),
    "PORT": ("server", "port"),
    "AUTH_PROVIDER": ("security", "auth_provider"),
    "REGISTRATION_MODE": ("security", "registration_mode"),
    "SECRET_KEY": ("security", "secret_key"),
    "CORS_ORIGINS": ("cors", "origins"),
    "DATA_DIR": ("storage", "data_dir"),
    "LANDSCAPE_URL": ("storage", "landscape_url"),
    "LANDSCAPE_PASSPHRASE": ("storage", "landscape_passphrase"),
    "PAYLOAD_STORE_PATH": ("storage", "payload_store_path"),
    "SESSION_DB_URL": ("storage", "session_db_url"),
    "COMPOSER_MODEL": ("composer", "model"),
    "COMPOSER_MAX_COMPOSITION_TURNS": ("composer", "max_composition_turns"),
    "COMPOSER_MAX_DISCOVERY_TURNS": ("composer", "max_discovery_turns"),
    "COMPOSER_TIMEOUT_SECONDS": ("composer", "timeout_seconds"),
    "COMPOSER_RATE_LIMIT_PER_MINUTE": ("composer", "rate_limit_per_minute"),
    "MAX_UPLOAD_BYTES": ("limits", "max_upload_bytes"),
    "MAX_BLOB_STORAGE_PER_SESSION_BYTES": ("limits", "max_blob_storage_per_session_bytes"),
    "ORPHAN_RUN_MAX_AGE_SECONDS": ("timeouts", "orphan_run_max_age_seconds"),
    "SERVER_SECRET_ALLOWLIST": ("security", "server_secret_allowlist"),
    # OIDC/Entra fields
    "OIDC_ISSUER": ("security", "oidc_issuer"),
    "OIDC_AUDIENCE": ("security", "oidc_audience"),
    "OIDC_CLIENT_ID": ("security", "oidc_client_id"),
    "OIDC_AUTHORIZATION_ENDPOINT": ("security", "oidc_authorization_endpoint"),
    "ENTRA_TENANT_ID": ("security", "entra_tenant_id"),
}
```

Unmapped `ELSPETH_WEB__*` env vars are rejected with a warning at startup (typo detection).

## Migration

### What happens to `WebSettings`?

Replaced by `ServerConfig`. Clean break, no wrapper. Callers change from
`app.state.settings.port` to `app.state.config.server.port`. No legacy code policy applies.

### What happens to `_settings_from_env()`?

Replaced by `load_server_config()` in `src/elspeth/web/config.py`:

```python
def load_server_config(
    config_path: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
) -> ServerConfig:
    """Load server config from file + env vars + CLI overrides.

    Precedence: CLI overrides > ELSPETH_WEB__* env vars > config file > defaults.
    """
```

### File location

Models and loading function stay in `src/elspeth/web/config.py`. This is a web concern,
not a core concern.

### What doesn't change

- TLS termination remains Caddy's responsibility. No TLS settings in `ServerConfig`.
- Pipeline execution config remains `ElspethSettings` loaded from pipeline YAML.
- The `ServerConfig` is for server process operation; `ElspethSettings` is for pipeline
  definition. They are intentionally separate systems.

## Not In Scope

- Orchestrator/engine configuration (separate activity)
- TLS termination (Caddy's job)
- Config contracts for ServerConfig (add if/when the web component gets its own
  runtime config layer — currently unnecessary since ServerConfig is used directly)
- Hot-reloading config on SIGHUP (nice-to-have, not MVP)

## Test Strategy

- Unit tests for each section model (defaults, validation, constraints)
- Unit tests for `load_server_config()` (file loading, env var merge, CLI override precedence)
- Unit tests for unknown YAML key rejection
- Unit tests for cross-section validators (secret_key + host, auth_provider + OIDC fields)
- Unit tests for `${VAR}` expansion in config values
- Unit tests for flat env var mapping
- Integration test: `elspeth web init` generates valid YAML that `load_server_config()` accepts
- Migration: existing `test_config.py` and `test_app.py` tests updated for new model paths
