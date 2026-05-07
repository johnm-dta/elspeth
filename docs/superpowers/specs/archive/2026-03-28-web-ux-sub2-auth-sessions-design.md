# Web UX Sub-Spec 2: Auth & Sessions

**Status:** Draft
**Date:** 2026-03-28
**Parent Spec:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`
**Phase:** 2
**Depends On:** Sub-Spec 1 (Foundation)
**Blocks:** Sub-Specs 4, 5, 6

---

## Scope

This sub-spec covers the authentication and session persistence layer for the
Web UX Composer MVP. It defines:

- The AuthProvider protocol and its three implementations (Local, OIDC, Entra).
- Auth middleware that extracts UserIdentity from Bearer tokens on protected routes.
- Auth routes for local login, token refresh, and user info.
- The SessionService protocol, SQLAlchemy table models, and CRUD operations.
- Session API routes including file upload with path traversal protection.
- Composition state versioning (immutable snapshots, monotonically increasing version per session).
- One active run per session enforcement.
- IDOR protection on all session-scoped routes.
- Session database schema creation on startup.

Out of scope: ComposerService integration (Phase 4), ExecutionService and run
lifecycle (Phase 5), frontend auth flows (Phase 6), Alembic migrations (W6 --
deferred to production readiness).

---

## AuthProvider Protocol

The AuthProvider protocol defines two methods that all auth implementations must
satisfy. It lives in `src/elspeth/web/auth/protocol.py` as a pure structural
interface with no exception definitions.

**Protocol methods:**

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| authenticate | async (token: str) -> UserIdentity | UserIdentity | AuthenticationError |
| get_user_info | async (token: str) -> UserProfile | UserProfile | AuthenticationError |

`authenticate` validates a token (JWT or IdP-issued) and returns the minimal
identity needed for request authorization. `get_user_info` returns the full
profile including display name, email, and group memberships.

**Auth models** (defined in `src/elspeth/web/auth/models.py`, not protocol.py):

**UserIdentity** -- frozen dataclass, slots=True. Minimal identity returned from
every auth check.

| Field | Type | Notes |
|-------|------|-------|
| user_id | str | Unique identifier from the auth provider |
| username | str | Human-readable username |

All fields are scalars. No freeze guard needed.

**UserProfile** -- frozen dataclass, slots=True. Extended profile.

| Field | Type | Notes |
|-------|------|-------|
| user_id | str | Matches UserIdentity.user_id |
| username | str | Matches UserIdentity.username |
| display_name | str | Human-friendly display name |
| email | str or None | Optional email address, default None |
| groups | tuple[str, ...] | Group memberships, default empty tuple |

All fields are scalars, None, or tuple of scalars. No freeze guard needed.

**AuthenticationError** -- exception class, defined in `auth/models.py`.

| Field | Type | Notes |
|-------|------|-------|
| detail | str | Human-readable error message, default "Authentication failed" |

Raised by all three providers when token validation fails. The auth middleware
catches this and converts it to HTTP 401.

---

## Auth Implementations

### LocalAuthProvider

File: `src/elspeth/web/auth/local.py`

Provides username/password authentication backed by a SQLite database. Passwords
are hashed with bcrypt. Tokens are JWTs signed with HMAC-SHA256 via
PyJWT.

**Constructor parameters:**

| Parameter | Type | Notes |
|-----------|------|-------|
| db_path | Path | Path to SQLite database file, created on first use |
| secret_key | str | HMAC signing key for JWTs |
| token_expiry_hours | int | JWT expiry in hours, default 24 |

**SQLite users table schema:**

| Column | Type | Constraint |
|--------|------|------------|
| user_id | TEXT | PRIMARY KEY |
| password_hash | TEXT | NOT NULL |
| display_name | TEXT | NOT NULL |
| email | TEXT | Nullable |

**Methods beyond the protocol:**

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| create_user | (user_id: str, password: str, display_name: str, email: str or None) -> None | None | ValueError if user exists |
| login | (username: str, password: str) -> str | JWT access token string | AuthenticationError("Invalid credentials") |

**JWT payload shape:**

| Claim | Type | Notes |
|-------|------|-------|
| sub | str | user_id |
| username | str | Same as user_id for local auth |
| exp | int | Unix timestamp, now + token_expiry_hours |

`authenticate` decodes and validates the JWT. On decode failure or expiry,
raises AuthenticationError("Invalid token"). On success, returns UserIdentity
with user_id and username from JWT claims.

`get_user_info` decodes the JWT, then queries the users table for display_name
and email. Returns UserProfile.

### OIDCAuthProvider

File: `src/elspeth/web/auth/oidc.py`

Validates tokens issued by any OIDC-compliant identity provider. The frontend
handles the IdP redirect; the backend only validates the resulting token.

**Constructor parameters:**

| Parameter | Type | Notes |
|-----------|------|-------|
| issuer | str | OIDC issuer URL (e.g. https://login.example.com) |
| audience | str | Expected `aud` claim value |
| jwks_cache_ttl_seconds | int | JWKS key cache TTL, default 3600 |

**Initialization behavior:**

On first `authenticate` call, fetches `{issuer}/.well-known/openid-configuration`
via httpx to discover the `jwks_uri`. Fetches the JWKS from that URI. Caches the
key set for `jwks_cache_ttl_seconds`. Subsequent calls reuse cached keys until
TTL expires.

**Token validation checks (in order):**

1. Decode JWT using cached JWKS (signature verification)
2. Verify `exp` claim (not expired)
3. Verify `iss` claim matches constructor `issuer`
4. Verify `aud` claim matches constructor `audience`

Any failure raises AuthenticationError with a descriptive detail message.

**Claim-to-identity mapping:**

| JWT Claim | Maps To |
|-----------|---------|
| sub | UserIdentity.user_id |
| preferred_username or sub | UserIdentity.username |
| name or preferred_username | UserProfile.display_name |
| email | UserProfile.email |
| groups (if present) | UserProfile.groups |

### EntraAuthProvider

File: `src/elspeth/web/auth/entra.py`

Wraps OIDCAuthProvider with Azure Entra ID specifics: tenant validation and
group claim extraction.

**Constructor parameters:**

| Parameter | Type | Notes |
|-----------|------|-------|
| tenant_id | str | Expected Azure tenant ID |
| audience | str | Application (client) ID |

**Derived behavior:**

- Sets OIDC issuer to `https://login.microsoftonline.com/{tenant_id}/v2.0`
- After standard OIDC validation, additionally verifies the `tid` claim matches
  `tenant_id`. Wrong tenant raises AuthenticationError("Invalid tenant").
- Extracts group memberships from the `groups` claim (list of group object IDs).
  Maps to UserProfile.groups as a tuple of strings.
- If `roles` claim is present, appends to UserProfile.groups with a `role:`
  prefix.

---

## Auth Middleware & Routes

### Auth Middleware

File: `src/elspeth/web/auth/middleware.py`

A FastAPI dependency function (not ASGI middleware) that extracts UserIdentity
from every protected request.

**Dependency signature:** `async def get_current_user(request: Request) -> UserIdentity`

**Behavior:**

1. Read `Authorization` header from request.
2. If missing or not `Bearer <token>` format, raise HTTPException(status_code=401, detail="Missing or invalid Authorization header").
3. Extract token string after "Bearer ".
4. Call `auth_provider.authenticate(token)` where auth_provider is retrieved from `request.app.state.auth_provider`.
5. On AuthenticationError, raise HTTPException(status_code=401, detail=exc.detail).
6. Return UserIdentity on success.

All session, catalog, execution, and composer routes declare this dependency via
`Depends(get_current_user)`.

### Auth Routes

File: `src/elspeth/web/auth/routes.py`

Router prefix: `/api/auth`

**POST /api/auth/login**

Available only when `WebSettings.auth_provider == "local"`. For OIDC/Entra
deployments, this endpoint returns 404.

Request body (JSON):

| Field | Type | Required |
|-------|------|----------|
| username | str | Yes |
| password | str | Yes |

Response (200):

| Field | Type | Notes |
|-------|------|-------|
| access_token | str | JWT token |
| token_type | str | Always "bearer" |

Error responses: 401 if credentials invalid.

**POST /api/auth/token**

Re-issues a JWT from a valid existing token. Available only for local auth.

Request: Bearer token in Authorization header (no body).

Response (200): Same shape as /login response -- new access_token with refreshed
expiry.

Error responses: 401 if token invalid or expired.

**GET /api/auth/config**

Returns the authentication configuration so the frontend can discover the auth
mode and OIDC redirect parameters at runtime. This endpoint is unauthenticated
(no Bearer token required) -- the frontend needs it before any login flow.

Response (200):

| Field | Type | Notes |
|-------|------|-------|
| provider | str | One of "local", "oidc", "entra" |
| oidc_issuer | str or null | Present when provider is "oidc" or "entra" |
| oidc_client_id | str or null | Present when provider is "oidc" or "entra" |

Values are read from `WebSettings.auth_provider`, `WebSettings.oidc_issuer`, and
`WebSettings.oidc_client_id`. The `oidc_client_id` field is added to WebSettings
alongside the existing OIDC fields (it is the OAuth 2.0 client ID the frontend
uses for the authorization code flow).

**GET /api/auth/me**

Returns the current user's profile. Available for all auth providers.

Request: Bearer token in Authorization header.

Response (200):

| Field | Type | Notes |
|-------|------|-------|
| user_id | str | |
| username | str | |
| display_name | str | |
| email | str or null | |
| groups | list[str] | |

Error responses: 401 if not authenticated.

---

## Data Model

All tables live in a dedicated session database (separate from the Landscape
audit database). SQLAlchemy Core table definitions in
`src/elspeth/web/sessions/models.py`. Schema creation via
`metadata.create_all(engine)` on application startup.

### sessions table

| Column | Type | Constraint |
|--------|------|------------|
| id | UUID | PRIMARY KEY, server-generated |
| user_id | VARCHAR | NOT NULL, indexed |
| title | VARCHAR | NOT NULL |
| created_at | DATETIME | NOT NULL, server default utcnow |
| updated_at | DATETIME | NOT NULL, server default utcnow, onupdate utcnow |

### chat_messages table

| Column | Type | Constraint |
|--------|------|------------|
| id | UUID | PRIMARY KEY, server-generated |
| session_id | UUID | NOT NULL, FOREIGN KEY -> sessions.id, indexed |
| role | VARCHAR | NOT NULL, CHECK IN ("user", "assistant", "system", "tool") |
| content | TEXT | NOT NULL |
| tool_calls | JSON | Nullable, LLM tool call records when role=assistant |
| created_at | DATETIME | NOT NULL, server default utcnow |

Ordered by created_at ascending for conversation history retrieval.

### composition_states table

| Column | Type | Constraint |
|--------|------|------------|
| id | UUID | PRIMARY KEY, server-generated |
| session_id | UUID | NOT NULL, FOREIGN KEY -> sessions.id, indexed |
| version | INTEGER | NOT NULL |
| source | JSON | Nullable, source plugin config |
| nodes | JSON | Nullable, list of transform/gate/aggregation specs |
| edges | JSON | Nullable, list of edge specs |
| outputs | JSON | Nullable, list of sink configs |
| metadata_ | JSON | Nullable, pipeline name, description |
| is_valid | BOOLEAN | NOT NULL, default False |
| validation_errors | JSON | Nullable, list of error strings |
| created_at | DATETIME | NOT NULL, server default utcnow |

**Constraints:**
- UNIQUE(session_id, version) -- enforces monotonic versioning per session.
- Rows are immutable once written. Each edit creates a new version.

### CompositionState Serialisation Contract

The `composition_states` table stores the pipeline graph as JSON fields with a
schema evolution envelope. The `CompositionState` frozen dataclass (defined in
sub-spec 4) represents the in-memory pipeline graph; it does NOT contain
`is_valid` or `validation_errors` -- those are separate DB columns on the
`composition_states` table, outside the serialised state.

**JSON envelope:** Each JSON column (`source`, `nodes`, `edges`, `outputs`,
`metadata_`) is stored with a `_version: int` key at the top level for schema
evolution. Version 1 is the initial format. Future schema changes increment this
version and add migration logic in `from_record()`.

**API serialisation:** `sessions/schemas.py` defines a `CompositionStateResponse`
Pydantic model that combines the deserialised `CompositionState` fields with
`is_valid` and `validation_errors` for API responses. This model is the only
shape returned by session state endpoints.

**Conversion functions** (part of the SessionService contract):

| Function | Signature | Purpose |
|----------|-----------|---------|
| `to_record` | (state: CompositionState, is_valid: bool, validation_errors: list[str] or None) -> dict | Serialises CompositionState fields to JSON-ready dict with `_version` envelope, plus `is_valid` and `validation_errors` as top-level keys for DB columns |
| `from_record` | (row: Row) -> CompositionStateRecord | Deserialises DB row back to CompositionStateRecord, validating `_version` and raising on unknown versions |

`to_record` and `from_record` are defined in `sessions/service.py` and are
internal to the SessionService implementation. They are not part of the
SessionServiceProtocol (callers never need to serialise/deserialise directly).

### runs table

| Column | Type | Constraint |
|--------|------|------------|
| id | UUID | PRIMARY KEY, server-generated |
| session_id | UUID | NOT NULL, FOREIGN KEY -> sessions.id, indexed |
| state_id | UUID | NOT NULL, FOREIGN KEY -> composition_states.id |
| status | VARCHAR | NOT NULL, CHECK IN ("pending", "running", "completed", "failed", "cancelled") |
| started_at | DATETIME | NOT NULL, server default utcnow |
| finished_at | DATETIME | Nullable |
| rows_processed | INTEGER | NOT NULL, default 0 |
| rows_failed | INTEGER | NOT NULL, default 0 |
| error | TEXT | Nullable, failure message |
| landscape_run_id | VARCHAR | Nullable, links to ELSPETH audit trail |
| pipeline_yaml | TEXT | Nullable, generated YAML that was executed |

**One active run per session (B6 fix):**

Application-level enforcement: before inserting a run with status "pending",
query for existing rows where session_id matches AND status IN ("pending",
"running"). If any found, raise RunAlreadyActiveError. This check-and-set
runs within the same database transaction as the insert.

For PostgreSQL deployments, additionally create a partial unique index:
`CREATE UNIQUE INDEX uq_runs_active_per_session ON runs(session_id) WHERE status IN ('pending', 'running')`.
This provides database-level enforcement as a safety net.

### run_events table

| Column | Type | Constraint |
|--------|------|------------|
| id | UUID | PRIMARY KEY, server-generated |
| run_id | UUID | NOT NULL, FOREIGN KEY -> runs.id, indexed |
| timestamp | DATETIME | NOT NULL, server default utcnow |
| event_type | VARCHAR | NOT NULL, CHECK IN ("progress", "error", "completed", "cancelled") |
| data | JSON | NOT NULL |

The data field shape depends on event_type:
- progress: `{"rows_processed": int, "rows_failed": int}`
- error: `{"message": str, "node_id": str or null, "row_id": str or null}`
- completed: `{"rows_processed": int, "rows_succeeded": int, "rows_failed": int, "rows_quarantined": int, "landscape_run_id": str}`
- cancelled: `{"rows_processed": int, "rows_failed": int}`

RunEvent doubles as the WebSocket payload model. Same shape whether forwarded
in-process (v1) or via Redis Streams (later extraction).

---

## SessionService

### Protocol

File: `src/elspeth/web/sessions/protocol.py`

**SessionServiceProtocol** -- runtime_checkable Protocol.

| Method | Signature | Returns |
|--------|-----------|---------|
| create_session | async (user_id: str, title: str) -> SessionRecord | SessionRecord |
| get_session | async (session_id: UUID) -> SessionRecord | SessionRecord |
| list_sessions | async (user_id: str) -> list[SessionRecord] | list[SessionRecord] |
| archive_session | async (session_id: UUID) -> None | None |
| add_message | async (session_id: UUID, role: str, content: str, tool_calls: dict or None) -> ChatMessageRecord | ChatMessageRecord |
| get_messages | async (session_id: UUID) -> list[ChatMessageRecord] | list[ChatMessageRecord] |
| save_composition_state | async (session_id: UUID, state: CompositionStateData) -> CompositionStateRecord | CompositionStateRecord |
| get_current_state | async (session_id: UUID) -> CompositionStateRecord or None | CompositionStateRecord or None |
| get_state | async (state_id: UUID) -> CompositionStateRecord | CompositionStateRecord |
| get_state_versions | async (session_id: UUID) -> list[CompositionStateRecord] | list[CompositionStateRecord] |
| set_active_state | async (session_id: UUID, state_id: UUID) -> CompositionStateRecord | CompositionStateRecord |
| create_run | async (session_id: UUID, state_id: UUID, pipeline_yaml: str or None = None) -> RunRecord | RunRecord |
| get_run | async (run_id: UUID) -> RunRecord | RunRecord |
| update_run_status | async (run_id: UUID, status: str, error: str or None = None, landscape_run_id: str or None = None, rows_processed: int or None = None, rows_failed: int or None = None) -> None | None |
| get_active_run | async (session_id: UUID) -> RunRecord or None | RunRecord or None |

`SessionRecord`, `ChatMessageRecord`, `CompositionStateRecord`, and `RunRecord` are frozen
dataclasses representing database rows. `CompositionStateData` is the input DTO
containing source, nodes, edges, outputs, metadata, is_valid, and
validation_errors.

### Implementation: SessionServiceImpl

File: `src/elspeth/web/sessions/service.py`

Uses SQLAlchemy Core with an async-compatible engine (aiosqlite for SQLite dev,
asyncpg for Postgres prod). Each method maps to one or more SQL operations
within a single transaction.

**Key behaviors:**

- `create_session`: generates UUID, inserts row, returns SessionRecord.
- `get_session`: selects by id. Raises ValueError if not found.
- `list_sessions`: selects where user_id matches, ordered by updated_at descending.
- `archive_session`: deletes the session and cascades to messages, states, runs, and events.
- `add_message`: inserts chat_messages row, updates session.updated_at.
- `get_messages`: selects where session_id matches, ordered by created_at ascending.
- `save_composition_state`: queries max version for session, increments by 1 (first version is 1), inserts new row. The UNIQUE(session_id, version) constraint prevents race conditions.
- `get_current_state`: selects the composition_states row with the highest version for the session. Returns None if no state exists.
- `get_state`: selects a composition_states row by its primary key (id). Raises ValueError if not found.
- `get_state_versions`: selects all composition_states for the session, ordered by version ascending.
- `set_active_state`: creates a new version record that is a copy of the specified prior version (looked up by state_id). The new record gets version = max(existing) + 1. This means "revert" always creates a new version, preserving full history. Execute and validate always use the latest version (from `get_current_state`). Raises ValueError if state_id not found or does not belong to the session.
- `create_run`: inserts a new runs row with status="pending", linking to the specified session and state. If `pipeline_yaml` is provided, stores the generated YAML at creation time. Enforces one-active-run per session (raises RunAlreadyActiveError if a pending or running run exists). The check-and-insert runs within a single transaction.
- `get_run`: selects a runs row by id. Raises ValueError if not found.
- `update_run_status`: updates the status (and optionally error, landscape_run_id, rows_processed, rows_failed) of a run. The optional parameters only update the column when not None. Sets finished_at to utcnow when status transitions to "completed", "failed", or "cancelled". Raises ValueError if run not found.
- `get_active_run`: selects the runs row for the session where status IN ("pending", "running"). Returns None if no active run exists.

---

## Session API

File: `src/elspeth/web/sessions/routes.py`

Router prefix: `/api/sessions`

All endpoints require authentication via `Depends(get_current_user)`.

### IDOR Protection (W5)

Every endpoint that takes a session_id path parameter must verify that
`session.user_id == current_user.user_id`. If the session belongs to a different
user, return 404 (not 403, to avoid leaking session existence). This check
applies to GET, DELETE, POST (messages, upload, validate, execute), and all
nested resource endpoints.

### Endpoints

**POST /api/sessions**

Creates a new session for the authenticated user.

Request body:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| title | str | No | Default: "New session" |

Response (201):

| Field | Type |
|-------|------|
| id | str (UUID) |
| user_id | str |
| title | str |
| created_at | str (ISO 8601) |
| updated_at | str (ISO 8601) |

**GET /api/sessions**

Lists sessions for the authenticated user. User-scoped: only returns sessions
where user_id matches the authenticated user.

Response (200): list of session objects (same shape as POST response).

**GET /api/sessions/{id}**

Returns a single session. IDOR-protected.

Response (200): session object.
Error: 404 if not found or belongs to another user.

**DELETE /api/sessions/{id}**

Archives (deletes) a session and all associated data. IDOR-protected.

Response (204): no body.
Error: 404 if not found or belongs to another user.

**POST /api/sessions/{id}/messages**

Sends a user message. In Phase 2, this only persists the message. In Phase 4,
this endpoint will be extended to trigger the ComposerService.

Request body:

| Field | Type | Required |
|-------|------|----------|
| content | str | Yes |

Response (200):

| Field | Type | Notes |
|-------|------|-------|
| message | ChatMessage | The persisted user message |
| state | CompositionState or null | null until Phase 4 |

**GET /api/sessions/{id}/messages**

Returns conversation history for a session. IDOR-protected.

Response (200): list of message objects:

| Field | Type |
|-------|------|
| id | str (UUID) |
| session_id | str (UUID) |
| role | str |
| content | str |
| tool_calls | object or null |
| created_at | str (ISO 8601) |

**GET /api/sessions/{id}/state**

Returns the current (highest-version) composition state. IDOR-protected.

Response (200): composition state object, or null if no state exists.

| Field | Type |
|-------|------|
| id | str (UUID) |
| session_id | str (UUID) |
| version | int |
| source | object or null |
| nodes | list or null |
| edges | list or null |
| outputs | list or null |
| metadata | object or null |
| is_valid | bool |
| validation_errors | list[str] or null |
| created_at | str (ISO 8601) |

**GET /api/sessions/{id}/state/versions**

Returns all composition state versions for a session. IDOR-protected.

Response (200): list of composition state objects, ordered by version ascending.

**GET /api/sessions/{id}/state/yaml**

Returns the generated YAML for the current composition state. IDOR-protected.
Defined in Sub-Spec 4 (Composer), wired here in `sessions/routes.py`. The route
handler loads the session's active CompositionState and calls
`generate_yaml(state)` (from `composer/yaml_generator.py`).

Response (200): `{"yaml": str}` — the YAML string for display in the frontend's
YAML tab.

Error: 404 if session not found, belongs to another user, or has no
CompositionState yet.

**POST /api/sessions/{id}/state/revert**

Reverts the pipeline to a prior composition state version. Creates a new version
that is a copy of the specified prior state (history is never rewritten).
IDOR-protected.

Request body:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| state_id | str (UUID) | Yes | ID of the prior CompositionState version to revert to |

Behavior:

1. Calls `session_service.set_active_state(session_id, state_id)` to create a
   new version that copies the specified prior state.
2. Injects a system message via
   `session_service.add_message(session_id, role="system", content="Pipeline reverted to version N.")`
   where N is the version number of the original state being reverted to.

Response (200): the new CompositionStateRecord (the copy at the new version
number). Same shape as `GET /api/sessions/{id}/state`.

Error responses:
- 404 if session not found or belongs to another user (IDOR protection).
- 404 if state_id not found or does not belong to this session.

**POST /api/sessions/{id}/upload**

Uploads a source file to the user's scratch directory. IDOR-protected.

Request: multipart/form-data with a `file` field.

Response (200):

| Field | Type | Notes |
|-------|------|-------|
| path | str | Server-side path to the saved file |
| filename | str | Original filename |
| size_bytes | int | File size |

Error responses:
- 413 if file exceeds `WebSettings.max_upload_bytes`.
- 404 if session not found or belongs to another user.

---

## File Upload

Upload destination: `{WebSettings.data_dir}/uploads/{sanitized_user_id}/{filename}`

### Path Traversal Sanitization (B5)

The user_id comes from the auth provider and is used to construct a directory
path. A compromised or malicious auth provider could supply a user_id containing
path traversal sequences (e.g., `../../etc`).

**Required sanitization:** `sanitized = Path(user_id).name`

`Path.name` extracts only the final component, stripping all directory
separators and parent references. `../../etc` becomes `etc`. An empty result
after sanitization (e.g., user_id of `..`) must raise a ValueError.

The filename from the uploaded file must also be sanitized with `Path(filename).name`
to prevent path traversal via crafted filenames.

The upload directory is created with `mkdir(parents=True, exist_ok=True)` on
first upload for each user.

File size is checked against `WebSettings.max_upload_bytes` before writing. The
check reads the file content into memory (acceptable for the 100MB default limit)
rather than trusting Content-Length headers.

---

## Security Constraints

### IDOR Protection (W5)

All session-scoped endpoints verify that the session's user_id matches the
authenticated user's user_id. Failure returns 404, not 403, to avoid revealing
whether a session ID exists for another user.

Implementation: a shared helper function `verify_session_ownership(session, user)`
called at the top of every session-scoped route handler, before any business
logic.

Test requirement: at least one test per endpoint group (session CRUD, messages,
state, upload) that creates a session as user A and attempts to access it as
user B, asserting a 404 response.

### One Active Run Per Session (B6)

Enforced at the SessionService level, not the route level. The service checks
for existing active runs before inserting a new one. This means the constraint
is enforced regardless of the caller (route handler, ComposerService, or
ExecutionService).

Raises `RunAlreadyActiveError` (a domain exception defined in
`sessions/service.py`). The route handler converts this to HTTP 409 Conflict.

### Secret Key Production Guard (W16, upgraded to S3)

On application startup, if `WebSettings.secret_key == "change-me-in-production"`
and the environment is not test (determined by checking for `pytest` in
`sys.modules` or an explicit `ELSPETH_ENV=test` environment variable), **raise
`SystemExit` with a clear error message**: `"FATAL: WebSettings.secret_key is
set to the default value. Set a secure secret_key before starting the web
server. See WebSettings documentation."` This is a hard crash, not a warning.

**Rationale (security fix S3):** The default `secret_key` is a well-known string.
Any deployment that fails to change it has zero authentication — an attacker can
forge valid JWTs for any user with zero skill. A warning is insufficient because
it can be missed in log output, and the system appears to function correctly with
the default key. A crash makes the failure impossible to ignore.

**Test environments** (where `pytest` is in `sys.modules` or `ELSPETH_ENV=test`)
are exempt because test fixtures need a predictable key for JWT generation in
test setup.

### OIDC/Entra Conditional Field Validation (from Sub-1 review)

`WebSettings` must validate that OIDC/Entra-specific fields are populated when their auth provider is selected. Add a `@model_validator(mode="after")` to `WebSettings` in `src/elspeth/web/config.py`:

- When `auth_provider="oidc"`: `oidc_issuer`, `oidc_audience`, and `oidc_client_id` must all be non-None. Raise `ValueError` if any are missing.
- When `auth_provider="entra"`: all three OIDC fields above must be non-None, AND `entra_tenant_id` must be non-None. Raise `ValueError` if any are missing.
- When `auth_provider="local"`: no conditional requirements (OIDC/Entra fields may be None).

This catches misconfiguration at construction time rather than at first auth request, producing a clear error instead of a `NoneType` traceback deep in the OIDC validation flow.

**Origin:** Sub-1 PR review (type-design-analyzer). Deferred to Sub-2 because Sub-1 has no auth middleware — validating fields before the consumer exists risks assumptions that may not hold. Tracked as `elspeth-34df5d61e4`.

### Session DB Schema Creation (W6)

On application startup, the app factory calls `metadata.create_all(engine)` to
ensure all tables exist. This is acceptable for development with SQLite. For
production PostgreSQL deployments, Alembic migrations should be used instead
(deferred -- not in this sub-spec).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/auth/__init__.py` | Module init |
| Create | `src/elspeth/web/auth/protocol.py` | AuthProvider protocol (two methods, no exceptions) |
| Create | `src/elspeth/web/auth/models.py` | UserIdentity, UserProfile, AuthenticationError |
| Create | `src/elspeth/web/auth/local.py` | LocalAuthProvider -- SQLite, bcrypt, JWT via PyJWT |
| Create | `src/elspeth/web/auth/oidc.py` | OIDCAuthProvider -- JWKS discovery, token validation, audience check |
| Create | `src/elspeth/web/auth/entra.py` | EntraAuthProvider -- tenant/group claims on top of OIDC |
| Create | `src/elspeth/web/auth/middleware.py` | get_current_user FastAPI dependency |
| Create | `src/elspeth/web/auth/routes.py` | /api/auth/login, /api/auth/token, /api/auth/me |
| Create | `src/elspeth/web/sessions/__init__.py` | Module init |
| Create | `src/elspeth/web/sessions/protocol.py` | SessionServiceProtocol |
| Create | `src/elspeth/web/sessions/models.py` | SQLAlchemy table definitions (sessions, chat_messages, composition_states, runs, run_events) |
| Create | `src/elspeth/web/sessions/service.py` | SessionServiceImpl -- CRUD, state versioning, active run check |
| Create | `src/elspeth/web/sessions/routes.py` | /api/sessions/* endpoints with IDOR protection |
| Create | `src/elspeth/web/sessions/schemas.py` | Pydantic request/response models for all session endpoints, including CompositionStateResponse |
| Modify | `src/elspeth/web/app.py` | Register auth and session routers, create session DB engine, call metadata.create_all on startup |
| Modify | `src/elspeth/web/dependencies.py` | Add get_current_user, get_session_service, get_auth_provider dependencies |
| Create | `tests/unit/web/auth/__init__.py` | Test package |
| Create | `tests/unit/web/auth/test_local_provider.py` | LocalAuthProvider: create user, login, authenticate, get_user_info, invalid credentials, invalid token |
| Create | `tests/unit/web/auth/test_oidc_provider.py` | OIDCAuthProvider: JWKS discovery (mocked httpx), valid/invalid/expired tokens, audience mismatch |
| Create | `tests/unit/web/auth/test_entra_provider.py` | EntraAuthProvider: valid Entra token, wrong tenant, group extraction |
| Create | `tests/unit/web/auth/test_middleware.py` | get_current_user: valid Bearer, missing header, invalid token |
| Create | `tests/unit/web/sessions/__init__.py` | Test package |
| Create | `tests/unit/web/sessions/test_service.py` | Session CRUD, message persistence, state versioning, active run enforcement |
| Create | `tests/unit/web/sessions/test_routes.py` | All session endpoints via TestClient, IDOR tests, upload path traversal test, file size limit test |

---

## Acceptance Criteria

### Auth Protocol and Models

1. AuthProvider protocol defines `authenticate` and `get_user_info` with correct signatures.
2. UserIdentity, UserProfile, and AuthenticationError are defined in `auth/models.py`, not `auth/protocol.py`.
3. UserIdentity and UserProfile are frozen dataclasses with slots=True.

### LocalAuthProvider

4. Creating a user and logging in with correct password returns a valid JWT.
5. Logging in with wrong password raises AuthenticationError("Invalid credentials").
6. `authenticate` with a valid JWT returns the correct UserIdentity.
7. `authenticate` with a garbage token raises AuthenticationError("Invalid token").
8. `authenticate` with an expired JWT raises AuthenticationError.
9. `get_user_info` returns a UserProfile with display_name and email from the SQLite database.

### OIDCAuthProvider

10. JWKS discovery fetches from `{issuer}/.well-known/openid-configuration` on first call.
11. JWKS is cached and reused until TTL expires.
12. Token with valid signature, issuer, audience, and expiry returns UserIdentity.
13. Token with wrong audience raises AuthenticationError.
14. Token with wrong issuer raises AuthenticationError.
15. Expired token raises AuthenticationError.

### EntraAuthProvider

16. Token with correct tenant ID (`tid` claim) passes validation.
17. Token with wrong tenant ID raises AuthenticationError("Invalid tenant").
18. Group object IDs from the `groups` claim are extracted into UserProfile.groups.

### Auth Middleware

19. Request with valid `Authorization: Bearer <token>` header returns UserIdentity.
20. Request with missing Authorization header returns HTTP 401.
21. Request with malformed Authorization header (not Bearer) returns HTTP 401.
22. Request with invalid token returns HTTP 401 with the AuthenticationError detail.

### Auth Routes

23. `POST /api/auth/login` with valid credentials returns `{access_token, token_type: "bearer"}`.
24. `POST /api/auth/login` with invalid credentials returns HTTP 401.
25. `POST /api/auth/login` returns 404 when auth_provider is not "local".
26. `POST /api/auth/token` with a valid Bearer token returns a new access_token.
27. `GET /api/auth/me` returns the full UserProfile for the authenticated user.
28a. `GET /api/auth/config` returns `{provider: "local"}` with null OIDC fields for local auth.
28b. `GET /api/auth/config` returns `{provider: "oidc", oidc_issuer: ..., oidc_client_id: ...}` for OIDC auth.
28c. `GET /api/auth/config` is accessible without authentication.

### SessionService

28. `create_session` generates a UUID and persists the session with the given user_id and title.
29. `get_session` returns the session for a valid ID, raises ValueError for unknown ID.
30. `list_sessions` returns only sessions belonging to the specified user_id, ordered by updated_at descending.
31. `archive_session` deletes the session and all associated messages, states, runs, and events.
32. `add_message` persists a chat message and updates the session's updated_at.
33. `get_messages` returns messages in created_at ascending order.
34. `save_composition_state` assigns version = max(existing versions) + 1, starting at 1.
35. `get_current_state` returns the highest-version state, or None if none exists.
36. `get_state_versions` returns all versions in ascending order.
37. The UNIQUE(session_id, version) constraint prevents duplicate versions.
37a. `get_state` returns a specific CompositionStateRecord by its UUID primary key.
37b. `set_active_state` creates a new version that is a copy of the specified prior version, with an incremented version number.
37c. `set_active_state` preserves full version history (revert does not delete or overwrite prior versions).
37d. `create_run` inserts a run with status="pending" and links it to the specified session and state.
37e. `get_run` returns the RunRecord for a valid run ID, raises ValueError for unknown ID.
37f. `update_run_status` updates the run status and sets finished_at when transitioning to a terminal status.
37g. `get_active_run` returns the pending/running run for a session, or None if no active run exists.

### Session Routes

38. `POST /api/sessions` creates a session scoped to the authenticated user.
39. `GET /api/sessions` returns only the authenticated user's sessions.
40. All session-scoped endpoints return 404 when accessed by a user who does not own the session (IDOR protection, W5).
41. `DELETE /api/sessions/{id}` returns 204 and cascades deletion.
42. `POST /api/sessions/{id}/messages` persists a user message and returns it.
43. `GET /api/sessions/{id}/messages` returns conversation history.
44. `GET /api/sessions/{id}/state` returns the current composition state or null.
45. `GET /api/sessions/{id}/state/versions` returns all state versions.

### File Upload

46. `POST /api/sessions/{id}/upload` saves the file to `{data_dir}/uploads/{sanitized_user_id}/{filename}`.
47. A user_id containing path traversal (`../../etc`) is sanitized to just the final component via `Path(user_id).name` (B5).
48. A filename containing path traversal is sanitized via `Path(filename).name`.
49. Files exceeding `WebSettings.max_upload_bytes` are rejected with HTTP 413.
50. The response includes the server-side path, original filename, and size in bytes.

### One Active Run Per Session

51. Inserting a "pending" run when a "pending" or "running" run exists for the same session raises RunAlreadyActiveError (B6).
52. The check-and-set runs within a single database transaction.
53. Completed, failed, and cancelled runs do not block new run creation.

### Schema Creation

54. Application startup calls `metadata.create_all(engine)` to create all session tables.
55. All five tables (sessions, chat_messages, composition_states, runs, run_events) exist after startup.
