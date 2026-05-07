# Web UX Composer MVP — Seam Contracts

**Status:** Draft
**Date:** 2026-03-28
**Parent:** `docs/superpowers/meta/web-ux-program.md`
**Purpose:** Define the exact type conversions, error shapes, and bridging patterns
at every inter-module boundary. Each seam has a single owner responsible for the
contract and the tests that verify it.

---

## Why This Document Exists

The six sub-specs define modules that are internally consistent but whose
inter-module handoff points were specified independently. This document makes the
seam contracts explicit so that:

1. Each boundary has an **owner** — one sub-spec is responsible for the contract
2. Each boundary has a **type conversion** — the exact function that transforms
   one module's output into another module's input
3. Each boundary has an **error shape** — the exact structure of error responses
4. Both sides of each seam have **tests** — the owner tests the contract, the
   consumer tests against it

---

## Seam A: CompositionState Lifecycle (Owner: Sub-4)

**Boundary:** Sub-4 (domain model) ↔ Sub-2 (persistence) ↔ Sub-5 (validation/execution) ↔ Sub-6 (rendering)

### Type Conversions

| Direction | Function | Location | Input | Output |
|-----------|----------|----------|-------|--------|
| Domain → JSON | `CompositionState.to_dict()` | `composer/state.py` | Frozen dataclass | Plain `dict` (MappingProxyType→dict, tuple→list) |
| JSON → Domain | `CompositionState.from_dict(d)` | `composer/state.py` | Plain `dict` | Frozen dataclass (deep-frozen via `__post_init__`) |
| Domain → DB | `to_record(state, is_valid, errors)` | `sessions/service.py` | `CompositionState` + validation | DB column dict with `_version` envelope |
| DB → Domain | `from_record(row)` | `sessions/service.py` | DB row | `CompositionState` (via `from_dict`) |
| Domain → API | `CompositionStateResponse.from_state(state, record)` | `sessions/schemas.py` | `CompositionState` + DB record | Pydantic model with `is_valid`, `validation_errors`, `id`, `session_id` |
| Domain → YAML | `generate_yaml(state)` | `composer/yaml_generator.py` | `CompositionState` | YAML string (via `to_dict()`) |

### Round-Trip Invariant

```
state == CompositionState.from_dict(state.to_dict())
```

Tested in `tests/unit/web/composer/test_state.py`.

### DB Column Name Mapping

| Dataclass field | DB column | Reason |
|-----------------|-----------|--------|
| `metadata` | `metadata_` | Avoids SQLAlchemy reserved name |

`to_record()` maps `metadata` → `metadata_`. `from_record()` maps `metadata_` → `metadata`. This mapping is internal to `sessions/service.py`.

### Who Calls What

| Caller | Conversion | Notes |
|--------|-----------|-------|
| `POST /messages` route handler | `from_record()` → pass `CompositionState` to `compose()` | Sub-4 route code |
| `POST /messages` route handler | `to_record()` from `ComposerResult.state` | Sub-4 route code |
| `POST /validate` route handler | `from_record()` → pass to `validate_pipeline()` | Sub-5 route code |
| `execute()` in ExecutionService | `from_record()` → `generate_yaml()` | Sub-5 service code |
| `GET /state` route handler | `from_record()` → `CompositionStateResponse` | Sub-2 route code |

### Owner Responsibilities

**Sub-4 owns:** `to_dict()`, `from_dict()`, `generate_yaml()`, and the round-trip test.
**Sub-2 owns:** `to_record()`, `from_record()`, `CompositionStateResponse`, and the DB round-trip test.

---

## Seam B: Composer ↔ Session Route (Owner: Sub-4)

**Boundary:** `POST /api/sessions/{id}/messages` route handler calls both
`SessionService` (Sub-2) and `ComposerService` (Sub-4).

### Parameter Types

```python
# ComposerService protocol
async def compose(
    message: str,
    messages: list[dict[str, Any]],  # pre-fetched chat history as LLM message dicts
    state: CompositionState,         # reconstructed from DB via from_dict()
) -> ComposerResult
```

**`messages` parameter:** The route handler calls `session_service.get_messages(session_id)`
to get `list[ChatMessageRecord]`, then converts each record to an LLM message dict
(`{"role": record.role, "content": record.content}`). ComposerService receives
pre-formatted LLM messages, not raw DB records. This keeps ComposerService
decoupled from SessionService's record types. ComposerService does NOT have a
dependency on SessionService — the route handler mediates.

**Rationale for `list[dict]` over `list[ChatMessageRecord]`:** The ComposerService
needs messages in LLM wire format (role + content dicts), not DB record format.
Converting in the route handler means the service has no knowledge of the
persistence layer's types. This was changed from the original `ChatMessageRecord`
design in Sub-4 Amendment 1.

### State Persistence After Compose

After `compose()` returns a `ComposerResult`, the route handler:

1. Calls `state.validate()` to get a `ValidationSummary` (is_valid, errors).
2. Calls `session_service.save_composition_state(session_id, CompositionStateData(
   source=result.state.to_dict()["source"],
   nodes=result.state.to_dict()["nodes"],
   ... ,
   is_valid=summary.is_valid,
   validation_errors=list(summary.errors) if summary.errors else None,
   ))`.
3. Persists the assistant message via `session_service.add_message()`.

This assembly is the route handler's responsibility (Sub-4 owns the route modification).

### Error Shapes

| Exception | HTTP Status | Response Body |
|-----------|-------------|---------------|
| `ComposerConvergenceError` | 422 | `{"error_type": "convergence", "detail": "...", "turns_used": int}` |
| LLM network/rate limit | 502 | `{"error_type": "llm_unavailable", "detail": "..."}` |
| LLM auth failure | 502 | `{"error_type": "llm_auth_error", "detail": "..."}` |
| Session not found / IDOR | 404 | `{"detail": "Session not found"}` |

**Note:** All error responses use `detail` (not `message`) as the human-readable
field, matching FastAPI's default `HTTPException` format. Sub-4's custom errors
include `error_type` as an additional field. The frontend checks `error_type`
first, falls back to `detail`.

---

## Seam C: Catalog ↔ Composer Tools (Owner: Sub-4)

**Boundary:** Composer discovery tools delegate to `CatalogService` protocol.

### Namespace Convention

| Context | Plugin type values | Example |
|---------|--------------------|---------|
| CatalogService protocol methods | Singular: `"source"`, `"transform"`, `"sink"` | `get_schema("source", "csv")` |
| REST URL path segments | Plural: `sources`, `transforms`, `sinks` | `GET /api/catalog/sources` |
| LLM tool definitions | Singular (matches protocol) | `get_plugin_schema("source", "csv")` |

The REST route handler in `catalog/routes.py` translates plural → singular
before calling `CatalogService.get_schema()`.

---

## Seam D: Execution ↔ Sessions — Run Lifecycle (Owner: Sub-5)

**Boundary:** `ExecutionServiceImpl._run_pipeline()` (sync background thread) calls
`SessionService` (async) for Run record management.

### Async/Sync Bridging Pattern (B8)

All calls from `_run_pipeline()` to async SessionService methods use:

```python
def _call_async(self, coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
```

The `self._loop` is obtained from `asyncio.get_running_loop()` inside the FastAPI
lifespan and passed to `ExecutionServiceImpl` at construction.

### Run State Transitions

```
pending → running → completed
                  → failed
                  → cancelled
```

| Transition | Triggered by | Method |
|------------|-------------|--------|
| → pending | `execute()` (async, main thread) | `session_service.create_run()` |
| pending → running | `_run_pipeline()` start (sync, worker thread) | `_call_async(session_service.update_run_status())` |
| running → completed | `_run_pipeline()` success | `_call_async(session_service.update_run_status())` |
| running → failed | `_run_pipeline()` except BaseException | `_call_async(session_service.update_run_status())` |
| running → cancelled | Orchestrator detects shutdown_event | `_call_async(session_service.update_run_status())` |

### Error Shape for RunAlreadyActiveError

`RunAlreadyActiveError` is defined in `sessions/service.py` (Sub-2). Both
`session_service.create_run()` (Sub-2, authoritative) and `execute()` (Sub-5,
pre-check for better errors) enforce the B6 constraint.

HTTP response: `409 Conflict` with `{"error_type": "run_already_active", "detail": "A run is already in progress for this session."}`.

---

## Seam E: WebSocket Progress (Owner: Sub-5)

**Boundary:** `ProgressBroadcaster` (Sub-5) → WebSocket handler (Sub-5) → Frontend
`executionStore` (Sub-6).

### RunEvent Wire Format

```json
{
  "run_id": "uuid-string",
  "timestamp": "2026-03-28T12:00:00Z",
  "event_type": "progress | error | completed | cancelled",
  "data": { ... }
}
```

Event type payloads are defined in Sub-5. The frontend MUST handle all four types.

### WebSocket Close Codes

| Code | Meaning | Frontend action |
|------|---------|-----------------|
| 1000 | Normal closure (run reached terminal state) | Do NOT reconnect. Poll `GET /api/runs/{id}` for final status. |
| 1006 | Abnormal closure (network drop, server restart) | Auto-reconnect with exponential backoff. |
| 1011 | Internal error (server-side failure) | Do NOT reconnect. Poll `GET /api/runs/{id}` for final status. |
| 4001 | Auth failure (invalid/expired token) | Do NOT reconnect. Call `authStore.logout()`. |

### Terminal Event Contract

When a run reaches a terminal state, `_run_pipeline()` broadcasts exactly one
terminal RunEvent (`completed` or `cancelled`), then the WebSocket handler sends
it and closes the connection with code 1000. If `_run_pipeline()` fails with an
exception, the done callback logs the failure but does NOT broadcast — the WebSocket
closes with code 1011 and the frontend polls REST for the failure details.

---

## Seam F: Validation Gate (Owner: Sub-6)

**Boundary:** Frontend `executionStore.validationResult` ↔ `sessionStore.compositionState` ↔ Execute button state.

### Invariants

1. The Execute button is enabled **only when** `validationResult.is_valid === true`
   AND `validationResult` was computed against the **current** `compositionState.version`.

2. `executionStore.clearValidation()` MUST be called when `compositionState` changes
   for **any reason**: composer response, state revert, session switch. The trigger
   is a change in `compositionState.version`, not the source of the change.

3. The `revertToVersion()` action in `sessionStore` MUST call
   `executionStore.clearValidation()` before updating `compositionState`.

### Stage 1 vs Stage 2 Rendering

| Data source | Type | Renderer |
|-------------|------|----------|
| `compositionState.validation_errors` | `string[]` | Simple list (no component attribution) |
| `executionStore.validationResult.errors` | `ValidationError[]` (with `component_id`, `suggestion`) | Per-component detail view with highlighting |

These are separate rendering code paths. Stage 1 errors appear in the Spec tab
as a summary. Stage 2 errors appear in the validation banner with full attribution.

---

## Seam G: Auth Config → Frontend Login (Owner: Sub-2)

**Boundary:** `GET /api/auth/config` (Sub-2) → `LoginPage` (Sub-6).

### Wire Format

```json
{
  "provider": "local | oidc | entra",
  "oidc_issuer": "string | null",
  "oidc_client_id": "string | null"
}
```

This endpoint is unauthenticated. Values come from `WebSettings`. The frontend
caches the response in memory for the session lifetime.

### Error Envelope Convention

All non-2xx responses across the entire API use this envelope:

```json
{
  "detail": "Human-readable error message (always present)",
  "error_type": "machine-readable discriminator (present on domain errors, absent on generic HTTP errors)"
}
```

FastAPI's default `HTTPException` produces `{"detail": "..."}`. Domain-specific
errors add `error_type`. The frontend checks `error_type` first (if present),
falls back to HTTP status code, then falls back to `detail` text.

---

## Seam H: Source Path Security (Owner: Sub-5)

**Boundary:** LLM tool-use loop (Sub-4) → YAML generation → Execution (Sub-5).

### Path Allowlist

Source plugin options containing `path` or `file` keys are restricted to paths
under `{WebSettings.data_dir}/uploads/`. Enforcement points:

| Layer | Location | Action on violation |
|-------|----------|---------------------|
| **Composition-time** (Sub-4) | `set_source()` tool | Return `ToolResult(success=False)` with error |
| **Validation-time** (Sub-5) | `validate_pipeline()` step 1 | Return `ValidationResult(is_valid=False)` with error |
| **Execution-time** (Sub-5) | `_run_pipeline()` pre-check | Fail the run before plugin instantiation |

The validation-time check is authoritative (cannot be bypassed by prompt injection).
The composition-time check is for LLM self-correction feedback.

---

## Cross-Cutting: `tool_calls` JSON Schema (Owner: Sub-4)

The `chat_messages.tool_calls` column (Sub-2) stores the raw LiteLLM tool call
format as a JSON array:

```json
[
  {
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "set_source",
      "arguments": "{\"plugin\": \"csv\", \"options\": {\"path\": \"...\"}}"
    }
  }
]
```

This is the format returned by LiteLLM's `response.tool_calls`. The frontend's
`MessageBubble` extracts `function.name` for display and optionally shows
`function.arguments` in a collapsible section. The arguments string is
JSON-encoded (it is a string containing JSON, not a parsed object).
