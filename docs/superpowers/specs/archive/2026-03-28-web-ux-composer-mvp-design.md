# Web UX — LLM Composer MVP

**Status:** Draft
**Date:** 2026-03-28
**Branch:** RC4.2-UX
**Relates to:** Semi-Autonomous Platform epic (`elspeth-rapid-ea33f5`)

---

## Overview

A web-based UX for composing, validating, and executing ELSPETH pipelines through
natural language. Chat-first interaction with a server-side LLM composer. Delivered
as the first usable increment towards the full Semi-Autonomous Platform described in
`docs/architecture/semi-autonomous/design.md`.

**Target users:** Small team (2-5 people), single-tenant deployment, basic auth.

**Core loop:** User describes a pipeline in chat → LLM composes it via tool calls →
user reviews the deterministic spec → validates (dry-run) → executes.

---

## Scope — What's In and What's Out

### In scope (v1)

- Chat-first pipeline authoring via server-side LLM composer
- Plugin catalog browsing (sources, transforms, sinks)
- Deterministic spec view with component linking (click to highlight relationships)
- Read-only DAG graph visualization (React Flow)
- Read-only YAML view (generated from composition state)
- Two-stage validation: composition-time + on-demand dry-run using real engine code
- Pipeline execution with deliberate "Execute" gate
- Lightweight WebSocket progress (row count + exceptions)
- Run history per session
- Pluggable auth: LocalAuthProvider, OIDCAuthProvider, EntraAuthProvider
- `elspeth web` CLI entry point
- `[webui]` optional install extra (named `webui` because `web` is already taken by the web scraping extra)

### Out of scope (later increments)

- Governance tiers and approval workflows
- Live preview execution (real sample rows through the engine)
- Graph editing (direct manipulation of the DAG)
- YAML import/paste
- Multi-tenant isolation
- Redis Streams telemetry
- Temporal workflow orchestration
- Worker pool isolation (preview/execution)
- Sealed PipelineArtifact with cryptographic hashing
- Methodology citation export
- Template library

---

## Architecture

### Approach: Modular Monolith

Single FastAPI process with explicit internal service boundaries. Each module
exposes a Python Protocol interface. Cross-module calls go through the Protocol.
When a module is extracted to a separate service, the Protocol stays — only the
transport changes (in-process call → HTTP client).

Pipeline execution runs in a background thread pool within the same process.

```text
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Application                                         │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │  REST Routes     │  │  WebSocket Route  │                  │
│  │  /api/chat/*     │  │  /ws/runs/{id}    │                  │
│  │  /api/catalog/*  │  │                   │                  │
│  │  /api/runs/*     │  │  (row count +     │                  │
│  │  /api/auth/*     │  │   exceptions)     │                  │
│  └────────┬────────┘  └────────┬──────────┘                  │
│           │                     │                              │
│  ─────────┼─────────────────────┼──────── Protocol Layer ──── │
│           │                     │                              │
│  ┌────────▼────────┐  ┌────────▼──────────┐                  │
│  │ CatalogService   │  │ ExecutionService  │                  │
│  │                  │  │                   │                  │
│  │ list_sources()   │  │ validate(spec)    │                  │
│  │ list_transforms()│  │ execute(spec)     │                  │
│  │ list_sinks()     │  │ get_status(id)    │                  │
│  │ get_schema(name) │  │ cancel(id)        │                  │
│  └─────────────────┘  └───────────────────┘                  │
│                                                               │
│  ┌─────────────────┐  ┌───────────────────┐                  │
│  │ ComposerService  │  │ SessionService    │                  │
│  │                  │  │                   │                  │
│  │ compose(prompt)  │  │ create_session()  │                  │
│  │ refine(msg)      │  │ get_session(id)   │                  │
│  │ get_state()      │  │ list_sessions()   │                  │
│  │                  │  │ get_history(id)   │                  │
│  │ Owns LLM loop   │  │                   │                  │
│  │ Calls Catalog +  │  │ Owns persistence  │                  │
│  │ Execution tools  │  │ (task DB)         │                  │
│  └─────────────────┘  └───────────────────┘                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ AuthService (pluggable — Local, OIDC, Entra)            │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Extraction path

Each module becomes a separate service in the full platform:

| v1 Module | Full Platform Service |
|-----------|----------------------|
| CatalogService | Plugin registry microservice |
| ComposerService | Conversation Service |
| ExecutionService | Preview Workers + Execution Workers |
| SessionService | Task DB service (Temporal integration point) |
| AuthService | API Gateway (OAuth2/OIDC) |

---

## Data Model

### Core Entities

**Session** — a pipeline authoring conversation.

| Field | Type | Notes |
|-------|------|-------|
| id | UUID | Primary key |
| user_id | str | From auth provider |
| title | str | User-editable or auto-generated |
| created_at | datetime | |
| updated_at | datetime | |

**ChatMessage** — a message in a session's conversation.

| Field | Type | Notes |
|-------|------|-------|
| id | UUID | Primary key |
| session_id | FK(Session) | |
| role | enum | user, assistant, system, tool |
| content | str | Message text |
| tool_calls | JSON | LLM tool calls (if role=assistant) |
| created_at | datetime | |

**CompositionState** — an immutable snapshot of pipeline configuration.

| Field | Type | Notes |
|-------|------|-------|
| id | UUID | Primary key |
| session_id | FK(Session) | |
| version | int | Monotonically increasing per session |
| source | SourceSpec | JSON — source plugin config |
| nodes | list[NodeSpec] | JSON — transforms, gates, aggregations |
| edges | list[EdgeSpec] | JSON — connections between nodes |
| outputs | list[OutputSpec] | JSON — sink configs |
| metadata | dict | Pipeline name, description |
| is_valid | bool | Stage 1 validation result |
| validation_errors | list[str] | Composition-time validation errors |
| created_at | datetime | |

Each edit creates a new version. State is immutable once written. This is the
future `PipelineArtifact.composition_state` binding point.

**Run** — a pipeline execution.

| Field | Type | Notes |
|-------|------|-------|
| id | UUID | Primary key |
| session_id | FK(Session) | |
| state_id | FK(CompositionState) | Bound to specific version |
| status | enum | pending, running, completed, failed, cancelled |
| started_at | datetime | |
| finished_at | datetime | Nullable |
| rows_processed | int | Updated during execution |
| rows_failed | int | Updated during execution |
| error | str | Nullable — failure message |
| landscape_run_id | str | Links to ELSPETH audit trail |
| pipeline_yaml | str | Generated YAML that was executed |

**RunEvent** — a progress event during execution.

| Field | Type | Notes |
|-------|------|-------|
| run_id | FK(Run) | |
| timestamp | datetime | |
| event_type | enum | progress, error, completed, cancelled |
| data | JSON | `{rows_processed, rows_failed}` or exception detail |

RunEvent is the WebSocket payload model. Same shape whether forwarded in-process
(v1) or via Redis Streams (later).

### Auth Models

**AuthProvider Protocol:**

```python
class AuthProvider(Protocol):
    async def authenticate(self, token: str) -> UserIdentity: ...
    async def get_user_info(self, token: str) -> UserProfile: ...
```

**Implementations:**

- `LocalAuthProvider` — username/password, SQLite-backed user table, JWT tokens
- `OIDCAuthProvider` — validates tokens from any OIDC-compliant IdP (JWKS discovery)
- `EntraAuthProvider` — Azure Entra ID specifics (tenant validation, group claims)

For OIDC/Entra: the frontend redirects to the IdP for login. The backend only
validates tokens. No login endpoint needed — the token comes from the IdP.

`LocalAuthProvider` exposes `POST /api/auth/login` and `POST /api/auth/token`
for local-only auth flows.

---

## REST API

### Sessions & Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions` | Create session |
| GET | `/api/sessions` | List user's sessions |
| GET | `/api/sessions/{id}` | Session detail |
| DELETE | `/api/sessions/{id}` | Archive session |
| POST | `/api/sessions/{id}/messages` | Send message (triggers composer) |
| GET | `/api/sessions/{id}/messages` | Chat history |
| GET | `/api/sessions/{id}/state` | Current composition state |
| GET | `/api/sessions/{id}/state/versions` | State version history |
| POST | `/api/sessions/{id}/upload` | Upload source file to user scratch dir |

`POST /api/sessions/{id}/messages` is the primary interaction endpoint. It:

1. Persists the user message
2. Forwards to ComposerService with current composition state
3. Runs the LLM tool-use loop (may take several seconds — the frontend shows
   a "composing" indicator during this time)
4. Persists the assistant response and any new composition state version
5. Returns `{message: ChatMessage, state: CompositionState | null}`

`POST /api/sessions/{id}/upload` accepts a multipart file upload. The file is
saved to the user's scratch directory (`{data_dir}/uploads/{user_id}/`) and the
response returns the server-side path. The user can then reference this path in
the chat, or the LLM composer can use it directly when configuring a source.

### Catalog

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/catalog/sources` | List available source plugins |
| GET | `/api/catalog/transforms` | List available transforms |
| GET | `/api/catalog/sinks` | List available sinks |
| GET | `/api/catalog/{type}/{name}/schema` | Plugin config schema |

Read-only. Wraps `PluginManager` discovery. Serializes Pydantic config schemas
for the frontend and LLM tools.

### Execution

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/{id}/validate` | Dry-run validation (Stage 2) |
| POST | `/api/sessions/{id}/execute` | Start execution (returns run_id) |
| GET | `/api/runs/{id}` | Run status |
| POST | `/api/runs/{id}/cancel` | Cancel run |
| GET | `/api/runs/{id}/results` | Run results summary |
| WS | `/ws/runs/{id}` | Live progress stream |

The WebSocket endpoint streams `RunEvent` payloads:
`{rows_processed: int, rows_failed: int, exceptions: list[str]}`.

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Local auth only — returns JWT |
| POST | `/api/auth/token` | Token refresh (local auth) |
| GET | `/api/auth/me` | Current user identity |

For OIDC/Entra providers, the frontend handles the IdP redirect flow.
The backend validates the resulting token via the AuthProvider protocol.

---

## LLM Composer

### Tool-Use Loop

The ComposerService drives a bounded tool-use loop:

```python
async def compose(
    self,
    message: str,
    session: Session,
    state: CompositionState,
) -> ComposerResult:
    messages = self._build_messages(session, state, message)
    tools = self._get_tool_definitions()

    for turn in range(self._max_turns):  # default 20
        response = await self._llm.chat(messages=messages, tools=tools)

        for call in response.tool_calls:
            result = self._execute_tool(call, state)
            state = result.updated_state
            messages.append(tool_result_message(call, result))

        if not response.tool_calls:
            # LLM produced a final text response
            return ComposerResult(
                message=response.content,
                state=state,
            )

        messages.append(assistant_message(response))

    raise ComposerConvergenceError(
        f"Composer did not converge after {self._max_turns} turns"
    )
```

### Composition Tools

**Discovery tools (read-only):**

| Tool | Description |
|------|-------------|
| `list_sources()` | Available source plugins with summaries |
| `list_transforms()` | Available transforms with summaries |
| `list_sinks()` | Available sinks with summaries |
| `get_plugin_schema(type, name)` | Full config schema for a plugin |
| `get_expression_grammar()` | Gate expression syntax reference |
| `get_current_state()` | Full current composition state |

**Mutation tools (state-changing):**

| Tool | Description |
|------|-------------|
| `set_source(source_spec)` | Set the pipeline source |
| `upsert_node(id, node_spec)` | Add or update a transform/gate/aggregation |
| `upsert_edge(edge_spec)` | Add or update an edge |
| `remove_node(id)` | Remove a node and its edges |
| `remove_edge(id)` | Remove an edge |
| `set_metadata(patch)` | Set pipeline name, description |

Every mutation tool validates the resulting state against the CatalogService and
returns:

```python
@dataclass(frozen=True)
class ToolResult:
    success: bool
    updated_state: CompositionState  # full state after mutation
    validation: ValidationSummary    # is_valid + errors
    affected_nodes: list[str]        # what changed
```

### System Prompt

The LLM receives:

1. **System message** — role definition, tool usage instructions, constraints
2. **Injected context** (each turn) — current composition state (serialized),
   current validation status, available plugin summary (cached)
3. **Chat history** — full conversation for the session
4. **User message** — the current request

### LLM Configuration

The composition model is configured at deployment, not hardcoded. Uses LiteLLM
(already in ELSPETH's stack) for provider abstraction. The composition model is
separate from any model used by pipeline LLM transforms.

---

## Validation Model

### Stage 1: Composition-Time (every tool call)

- Plugin exists in catalog
- Config fields match plugin Pydantic schema
- Edge references valid node IDs
- Source is set
- At least one sink exists

Runs automatically during the LLM tool-use loop. Errors are returned as tool
results so the LLM can self-correct.

### Stage 2: On-Demand Dry-Run (user clicks Validate)

All of Stage 1, plus:

1. Generate ELSPETH YAML from CompositionState
2. Build `ExecutionGraph.from_plugin_instances()` — real engine code
3. Run `graph.validate()` — real engine code
   - Schema compatibility (source output → transform input)
   - Route destination validation (gates → real sinks/nodes)
   - Error sink coverage
   - Quarantine destination validation
4. Plugin instantiation check (can plugins be created with this config?)
5. Schema flow analysis (trace field types through the full DAG)

Results include per-component error attribution with suggested fixes.

### Execute Gate

The Execute button is disabled until Stage 2 passes. This is the deliberate
"really mean it" gate.

In a later increment, governance tiers insert between "valid" and "executable":
an approval banner and disabled button until sign-off. The UI change is minimal.

---

## Frontend UX

### Layout

Three-panel layout:

1. **Sessions sidebar** (left, 200px) — session list, new session button
2. **Chat panel** (centre, flex) — conversation with the LLM composer
3. **Inspector panel** (right, 320px) — tabbed view of the current pipeline

### Composing Indicator

While the backend is running the LLM tool-use loop, the chat panel shows a
visual "composing" indicator (typing dots animation or spinner with "ELSPETH is
thinking..." label). The send button is disabled during this time. This prevents
the user from sending overlapping requests and makes it clear the system is
working.

### Inspector Tabs

| Tab | Content | Interactivity |
|-----|---------|---------------|
| **Spec** | Deterministic technical specification | Click components to highlight upstream/downstream relationships |
| **Graph** | React Flow DAG visualization | Read-only in v1, pan/zoom |
| **YAML** | Generated ELSPETH pipeline YAML | Read-only, copyable |
| **Runs** | Execution history for this session | Click run for progress detail |

### Component Linking (Spec Tab)

Each component card in the spec view shows:

- Node type badge (SOURCE, TRANSFORM, GATE, SINK) with colour coding
- Config summary (plugin name, key settings)
- Next-node indicator: `↓ next_node_name` for default flow
- Route indicators for gates: `✓ true → sink_a`, `✗ false → sink_b`
- Error sink indicator: `⚠ on_error → error_sink`
- Fork path labels for fork nodes

**Click behaviour:**

- Click a component → it highlights with a SELECTED badge
- Direct upstream connections get an INPUT badge
- Direct downstream connections get OUTPUT / route-path badges (TRUE PATH, FALSE PATH)
- Unconnected components dim to 35% opacity
- Click again or click empty space to deselect

Linking data is derived from the `ExecutionGraph` edges — same source of truth as
the Graph tab and the actual engine execution.

### Execution Progress

When a run is in progress, the Runs tab shows:

- Status indicator (running/completed/failed)
- Progress bar (rows processed / estimated total)
- Three counters: rows processed, rows failed, estimated total
- Recent exceptions list (scrolling, most recent first)
- Cancel button

Data streams over the WebSocket connection.

### Validation Results

Shown inline in the inspector when the user clicks Validate:

- **Pass:** green banner with check details (node count, schema compatibility,
  route validation). Execute button enables.
- **Fail:** red banner with per-component errors and suggested fixes. Execute
  button disabled.

---

## Project Structure

### Source Layout

```text
src/elspeth/web/                    L3 (application layer)
├── __init__.py
├── app.py                          FastAPI application factory
├── config.py                       Web-specific settings
├── dependencies.py                 FastAPI dependency injection
│
├── auth/                           AuthService module
│   ├── protocol.py                 AuthProvider protocol
│   ├── local.py                    LocalAuthProvider
│   ├── oidc.py                     OIDCAuthProvider
│   ├── entra.py                    EntraAuthProvider
│   ├── middleware.py               FastAPI auth middleware
│   └── models.py                   UserIdentity, UserProfile
│
├── catalog/                        CatalogService module
│   ├── protocol.py                 CatalogService protocol
│   ├── service.py                  Wraps PluginManager
│   ├── routes.py                   GET /api/catalog/*
│   └── schemas.py                  API response models
│
├── composer/                       ComposerService module
│   ├── protocol.py                 ComposerService protocol
│   ├── service.py                  LLM tool-use loop
│   ├── tools.py                    Tool definitions
│   ├── prompts.py                  System prompt + context injection
│   └── state.py                    CompositionState, NodeSpec, EdgeSpec
│
├── execution/                      ExecutionService module
│   ├── protocol.py                 ExecutionService protocol
│   ├── service.py                  Wraps Orchestrator
│   ├── routes.py                   Execute, status, cancel, WS
│   ├── validation.py               Dry-run (calls real ExecutionGraph)
│   ├── progress.py                 WebSocket broadcaster
│   └── schemas.py                  RunStatus, RunEvent, ValidationResult
│
├── sessions/                       SessionService module
│   ├── protocol.py                 SessionService protocol
│   ├── service.py                  SQLAlchemy persistence
│   ├── routes.py                   Session/message CRUD
│   ├── models.py                   SQLAlchemy table models
│   └── schemas.py                  API request/response models
│
└── frontend/                       React SPA
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── App.tsx
        ├── main.tsx
        ├── api/                    Auto-generated API client
        ├── components/
        │   ├── chat/               ChatPanel, MessageBubble, ChatInput
        │   ├── inspector/          InspectorPanel, SpecView, GraphView
        │   ├── sessions/           SessionSidebar
        │   ├── execution/          ProgressView, RunHistory
        │   └── common/             Layout, AuthGuard
        ├── hooks/                  useSession, useComposer, useWebSocket
        ├── stores/                 Zustand (session, composition, auth)
        └── types/                  Generated from OpenAPI
```

### Layer Compliance

`web/` is L3 (application layer). It may import from L0 (`contracts/`),
L1 (`core/`), L2 (`engine/`), and L3 (`plugins/`). Nothing in
`engine/`, `core/`, or `contracts/` may import from `web/`.

Enforced by the existing `enforce_tier_model.py` CI script.

### Packaging

Optional install extra: `uv pip install -e ".[webui]"`

Adds: `fastapi`, `uvicorn`, `PyJWT[crypto]`, `httpx`,
`python-multipart`, `websockets`.

CLI entry point: `elspeth web [--port PORT] [--auth {local,oidc,entra}]`

Registered as a Typer subcommand in `cli.py`.

### Development

- Backend: `uvicorn elspeth.web.app:create_app --factory --reload`
- Frontend: `cd src/elspeth/web/frontend && npm run dev` (Vite dev server with
  proxy to FastAPI)
- Production: FastAPI serves built SPA from `frontend/dist/` as static files

---

## Avoid-Rework Decisions

These decisions are specifically chosen so that the v1 code becomes the real
production code when the full Semi-Autonomous Platform is built, rather than
throwaway prototyping.

| Decision | Why | Extraction Impact |
|----------|-----|-------------------|
| Protocol interface per module | Cross-module calls use Protocol, not direct imports | Swap in-process impl for HTTP client — same Protocol |
| CompositionState is immutable + versioned | Matches PipelineArtifact lineage model | Becomes `artifact.composition_state` |
| Run binds to specific CompositionState version | Not "latest" — exact state was executed | Becomes `artifact_hash` binding |
| RunEvent as WebSocket payload model | Same shape in-process or via Redis Streams | No format change on extraction |
| SQLAlchemy for persistence | Same ORM as Landscape, SQLite→Postgres is config | Models move with extracted service |
| Real engine validation in dry-run | Calls `ExecutionGraph.from_plugin_instances()` | No parallel validation to maintain |
| Composition tools match semi-autonomous API | `set_source`, `upsert_node`, `upsert_edge` | Become the Composition API |
| OpenAPI-generated TypeScript types | FastAPI auto-generates schema | No manual type duplication |
| AuthProvider as Protocol | Pluggable from day one | Moves to API Gateway |
| Component linking from ExecutionGraph edges | Same data as graph tab and engine | Maps to approval scope hashing |

---

## Relationship to Semi-Autonomous Platform Design

This spec implements a subset of the Semi-Autonomous Platform
(`docs/architecture/semi-autonomous/design.md`). The mapping:

| Semi-Autonomous Concept | v1 Implementation |
|--------------------------|-------------------|
| Conversation Service | ComposerService + SessionService |
| Composition API | Composer tools (discovery + mutation) |
| Declarative CompositionState | `CompositionState` entity (immutable, versioned) |
| Deterministic Spec | Spec tab in inspector panel |
| Summary-First Interaction | Chat panel is primary, graph is secondary |
| PipelineArtifact | CompositionState version + Run record (governance fields added later) |
| Static Security Policy | Not in v1 — Execute gate is validation-only |
| Preview Workers | Not in v1 — dry-run validation only |
| Execution Workers | In-process background thread (extraction to workers later) |
| Temporal Workflows | Not in v1 — SessionService is the integration point |
| Redis Streams | Not in v1 — in-process WebSocket forwarding |
| API Gateway | AuthService with pluggable providers |
| Graph Editor | Read-only React Flow in v1 |

---

## Open Questions

These questions were resolved during design review.

1. **Streaming assistant responses.** Wait for the full tool-use loop to complete.
   The chat panel shows a visual "composing" indicator (typing animation or
   spinner) while the backend processes the LLM loop, so the user knows the
   system is working. No token-by-token streaming in v1.

2. **Session sharing.** Sessions are user-scoped in v1. Each user sees only their
   own sessions.

3. **Source file upload.** Both modes supported: reference a server-accessible path,
   or upload a file through the web UI. Uploads go to a per-user scratch directory
   on the server (`{data_dir}/uploads/{user_id}/`). The scratch directory is
   managed by the SessionService. The source plugin receives the resolved server
   path regardless of how the file was provided.

4. **Concurrent executions.** One active run per session. If higher concurrency is
   needed, deploy additional instances per user rather than multiplexing runs
   within a single session.
