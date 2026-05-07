# Web UX Sub-Spec 4x: Composer Hardening

**Status:** Draft
**Date:** 2026-03-29
**Depends On:** Sub-Plan 4 (Composer) -- must be merged first
**Blocks:** Nothing (can run in parallel with Sub-Plan 5)
**Issues:** elspeth-7f9ecad8f7, elspeth-e957e00354, elspeth-5b3c9a7355, elspeth-519a32221d, elspeth-0a1dae9f90

---

## Overview

Five post-review findings from the Sub-Plan 4 Composer, plus eight cross-cutting
hardening fixes from multi-reviewer feedback. One P1 bug (partial state loss on
convergence failure), three P2 issues (turn budget scaling, rate limiting not
enforced, edge insertion order), and one P3 hygiene task (tool dispatcher pattern).
Hardening additions: discovery cache scoping (local variable, not instance field),
`get_current_state` cache exclusion, convergence handler persistence/validation
guards, rate limiter `Depends()` wiring and per-user locks, budget-at-classification
loop guard, server timeout ordering, `freeze_fields` on exception dataclass, and
multi-worker documentation. All changes are scoped to the `web/composer/` module
and its route handler; none require changes to Sub-Plans 1-3 or the core engine.

---

## Finding 1: Dynamic Turn Budget (elspeth-7f9ecad8f7)

### Problem

The composer loop uses a fixed `max_turns` (default 20) for both discovery and
composition. The Systems Thinker review identified this as a Limits to Growth
archetype: simple pipelines converge in 4-5 turns, but complex pipelines requiring
6 discovery calls + 8 composition calls + 3 validation retries = 17 turns, leaving
3 turns for refinement. The growth engine (pipeline complexity) runs into a fixed
ceiling (turn budget), and the system degrades ungracefully -- it just fails with
`ComposerConvergenceError`.

### Recommended Approach: Discovery Caching + Separate Composition Budget

The issue description lists four options. The correct answer is the hybrid (option
4), but with a specific decomposition:

**Discovery caching** eliminates the root cause. Discovery tool results (plugin
lists, schemas) are static within a session. The LLM re-calls `list_sources` and
`get_plugin_schema` because it cannot perfectly retain prior tool results across
turns. Caching these at the `ComposerServiceImpl` level means repeated discovery
calls return instantly from cache and do not consume an LLM turn -- they are
resolved locally before the LLM call.

**Separate budgets** address the residual risk. After caching, discovery calls no
longer consume turns, but the composition budget should still scale with complexity.
The fix: replace the single `composer_max_turns` with two settings:

| Setting | Default | Purpose |
|---------|---------|---------|
| `composer_max_composition_turns` | 15 | LLM round-trips that involve mutation tool calls |
| `composer_max_discovery_turns` | 10 | LLM round-trips that involve only discovery tool calls |

A turn is classified based on the tool calls it contains. If a turn contains at
least one mutation tool call, it counts against the composition budget. If it
contains only discovery tool calls, it counts against the discovery budget. A turn
with no tool calls terminates the loop (existing behaviour).

Why not dynamic scaling (option 2)? Dynamic scaling (`base + N * node_count`)
couples the turn budget to pipeline size, which creates a new problem: the budget
grows without bound as the pipeline grows. A pipeline with 50 nodes would get 65+
turns, each costing real money. Fixed budgets with caching are predictable and
auditable.

### Specific Changes

**`ComposerServiceImpl.__init__`:** Accept `max_composition_turns` and
`max_discovery_turns` from settings instead of `max_turns`.

**`ComposerServiceImpl.compose`:** Replace the single `range(self._max_turns)` loop
with a `while True` loop that checks budget exhaustion at classification time
after each turn. Track `composition_turns_used` and `discovery_turns_used`
separately. Do NOT use a conjunctive `while composition < max AND discovery < max`
guard -- that exits the loop when either budget is exhausted, even if the other
budget still has capacity and the current turn would charge the non-exhausted
budget. Instead, after classifying each turn's tool calls, check whether the
budget that would be charged is exhausted. Exit only when the specific budget
being charged has no remaining capacity. On convergence failure, report which
budget was exhausted.

**`_CACHEABLE_DISCOVERY_TOOLS`:** A `frozenset` at module level containing only:
`list_sources`, `list_transforms`, `list_sinks`, `get_plugin_schema`,
`get_expression_grammar`. `get_current_state` is explicitly excluded -- it returns
live state that changes with every mutation, so caching it would return stale
snapshots. `get_current_state` remains in `_DISCOVERY_TOOLS` for budget
classification purposes (it is a discovery turn, not a composition turn), but the
cache check uses `_CACHEABLE_DISCOVERY_TOOLS` membership, not `_DISCOVERY_TOOLS`.

**Discovery cache as local variable:** The cache is a `dict[str, Any]` local
variable allocated at the top of `compose()`, keyed by a cache key derived from
the tool name + arguments. It is threaded through helper methods as a parameter
rather than stored as `self._discovery_cache`. This eliminates the
concurrent-request race condition that would exist with an instance field -- two
concurrent `compose()` calls on the same service instance each get their own
independent cache dict. The cache is inherently per-request since it is a local
variable scoped to the `compose()` call.

**Discovery cache resolution:** Before calling the LLM, the service does NOT
intercept tool calls. After the LLM responds with tool calls, before executing
each tool, the service checks `_CACHEABLE_DISCOVERY_TOOLS` membership. If the
tool is cacheable and the cache key exists, the cached result is returned as the
tool message and neither budget counter is incremented. If the tool is cacheable
but not yet cached, the result is executed, cached, and the discovery counter is
incremented. `get_current_state` calls always execute (not cacheable) but still
count against the discovery budget. The LLM still "sees" all results as normal
tool responses.

**`ComposerConvergenceError`:** Add `budget_exhausted: str` field ("composition"
or "discovery") so the error message tells the user which limit was hit. The HTTP
422 response includes this field.

**WebSettings:** Replace `composer_max_turns: int = 20` with
`composer_max_composition_turns: int = 15` and
`composer_max_discovery_turns: int = 10`. Total possible turns = 25, but in
practice caching means discovery turns are rarely consumed. Add
`composer_timeout_seconds: float = 85.0` -- this must be strictly less than the
frontend's 90-second timeout so the server returns a structured error before the
frontend's fetch aborts. The timeout is enforced via `asyncio.wait_for()` around
the compose loop. On timeout, raise `ComposerConvergenceError` with
`budget_exhausted="timeout"`. **Timeout ordering:** frontend (90s) > server
compose timeout (85s) > individual LLM call timeout (set by LiteLLM, typically
30s). The server timeout catches runaway loops; the frontend timeout catches
server hangs. Multi-worker note: rate limiting is per-process; deployments with
multiple uvicorn workers require a shared store (e.g., Redis) for accurate
cross-process rate limiting.

**Backward compatibility:** Not applicable (CLAUDE.md: no legacy code policy).
Remove `composer_max_turns` entirely.

---

## Finding 2: Partial State Preservation on Convergence Failure (elspeth-e957e00354)

### Problem

When the composer hits `ComposerConvergenceError`, the route handler catches it
and returns HTTP 422 with an error body. The `CompositionState` that was
incrementally built across 15+ turns of successful tool calls is discarded. The
user loses all partial work.

This is the P1 issue. A user who asks for a complex pipeline and waits 30+ seconds
for the composer to work gets nothing back -- not even the 80% of the pipeline that
was successfully built.

### Current Behaviour

```
compose() loop:
  turn 1: set_source -> state v2
  turn 2: upsert_node -> state v3
  ...
  turn 15: upsert_edge -> state v16
  turn 16: validation error, LLM tries to fix
  ...
  turn 20: max_turns exceeded
  raise ComposerConvergenceError(20)  # state v16 is on the stack, unreturned

Route handler:
  except ComposerConvergenceError:
    return 422  # no state in the response
```

### Fix

**`ComposerConvergenceError`:** Add a `partial_state: CompositionState | None`
field. When `partial_state` is not `None`, `__post_init__` must call
`freeze_fields(self, "partial_state")` to satisfy the frozen dataclass
immutability contract (CLAUDE.md: deep_freeze). The composer loop tracks the last
state where `version > initial_version` (i.e., at least one successful mutation
occurred). On convergence failure, this state is attached to the exception.

**`ComposerServiceImpl.compose`:** Before raising `ComposerConvergenceError`,
set `partial_state` to the current `state` variable if it differs from the input
state (version changed). If no mutations occurred, `partial_state` is `None`.

```python
# In the compose loop, after exhausting turns:
partial = state if state.version > initial_version else None
raise ComposerConvergenceError(
    max_turns=self._max_turns,
    partial_state=partial,
    budget_exhausted=budget_exhausted,
)
```

**Route handler:** On `ComposerConvergenceError`, check `exc.partial_state`. If
present:
1. Run `partial_state.validate()` to get the validation summary.
2. Persist the partial state as a new version via `session_service.save_composition_state()`.
3. Return HTTP 422 with both the error body AND the partial state in the response.

```python
except ComposerConvergenceError as exc:
    response_body = {
        "error_type": "convergence",
        "detail": str(exc),
        "turns_used": exc.max_turns,
        "budget_exhausted": exc.budget_exhausted,
    }
    if exc.partial_state is not None:
        # Validate guard: if validation itself fails, persist with
        # is_valid=False rather than losing the state entirely.
        try:
            summary = exc.partial_state.validate()
        except Exception:
            summary = ValidationSummary(is_valid=False, errors=["validation_failed"])

        # Persistence guard: if save fails, log the error and return
        # 422 without partial_state rather than crashing to 500.
        try:
            await session_service.save_composition_state(
                session_id,
                CompositionStateData.from_state(exc.partial_state, summary),
            )
            # Persistence succeeded: partial state is now the current
            # version (advances the session state). Include it in the
            # response so the frontend can display the partial pipeline.
            response_body["partial_state"] = CompositionStateResponse.from_state(
                exc.partial_state, summary
            ).dict()
        except Exception:
            slog.error(
                "convergence_partial_state_save_failed",
                session_id=session_id,
                exc_info=True,
            )
            # Do NOT include partial_state in response -- it was not
            # persisted, so the frontend cannot resume from it.
    raise HTTPException(status_code=422, detail=response_body)
```

**Frontend impact (Sub-6):** The frontend already handles 422 convergence errors.
It needs to check for `partial_state` in the error body and, if present, update
`sessionStore.compositionState` so the inspector shows the partial pipeline. This
is a minor Sub-6 change but does not block this phase -- the backend change is
independently correct.

### Data Model Implications

`ComposerConvergenceError` gains two fields (`partial_state`, `budget_exhausted`).
`budget_exhausted` is one of `"composition"`, `"discovery"`, or `"timeout"`. When
`partial_state` is not `None`, `__post_init__` calls
`freeze_fields(self, "partial_state")` per the frozen dataclass immutability
contract. This is a breaking change to the exception class, which is fine -- no
legacy code policy applies. The 422 error body gains an optional `partial_state`
field and a `budget_exhausted` field. The seam contracts document (Seam B) must be
updated.

---

## Finding 3: Rate Limiting (elspeth-5b3c9a7355)

### Problem

`WebSettings.composer_rate_limit_per_minute` exists (added in R4 fix H8p) but is
not enforced. The `POST /api/sessions/{id}/messages` endpoint has no rate limiting.
A runaway client or rapid user can exhaust the LLM API budget.

### Recommended Approach: In-Memory Per-User Counter with FastAPI Dependency

Not pyrate-limiter. The existing `core/rate_limit/limiter.py` wraps pyrate-limiter
with SQLite persistence for cross-process rate limiting of pipeline source/sink
calls. That is heavyweight machinery for a web endpoint serving 2-5 users. The
correct tool here is a simple in-memory sliding window counter.

**Why not pyrate-limiter:** It has a background leaker thread, SQLite persistence
overhead, and a known race condition on cleanup (see the custom excepthook in
`core/rate_limit/limiter.py`). All of that complexity is justified for
pipeline-level rate limiting where persistence matters across process restarts. For
a web endpoint, an in-memory counter that resets on restart is sufficient. If the
server restarts, the rate limit resets -- that is fine.

**Why per-user, not per-session:** Per-session limiting lets a user create 10
sessions and send 10x the intended rate. Per-user limiting matches the intent: "a
single human user should not send more than N messages per minute." The user
identity comes from the auth middleware (`request.state.user.id`).

### Implementation

**`src/elspeth/web/middleware/rate_limit.py`:** New module. A FastAPI dependency
(not middleware -- dependencies are per-route, middleware is global).

```python
class ComposerRateLimiter:
    """In-memory sliding window rate limiter for composer messages.

    Tracks message timestamps per user_id. On each request, prunes
    timestamps older than 60 seconds, then checks count against limit.
    Returns 429 if exceeded.

    Uses per-user asyncio.Lock instances to avoid contention between
    unrelated users. asyncio.Lock guards coroutine suspension points
    (e.g., between the prune-check-append sequence where another
    coroutine could interleave), not thread safety. A top-level
    _locks_lock (held for microseconds -- dict lookup only) serializes
    creation/fetch of per-user locks.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._buckets: dict[str, list[float]] = {}
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a lock for the given user."""
        async with self._locks_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    async def check(self, user_id: str) -> None:
        """Raise HTTPException(429) if rate limit exceeded."""
        lock = await self._get_user_lock(user_id)
        async with lock:
            ...
```

**Route wiring via `Depends()`:** The rate limiter is wired as a FastAPI
dependency, matching the existing `Depends(get_current_user)` pattern. The
`ComposerRateLimiter` instance is created in the app factory lifespan using
`settings.composer_rate_limit_per_minute` and stored on `app.state`. A
`get_rate_limiter` dependency function extracts it:

```python
async def get_rate_limiter(request: Request) -> ComposerRateLimiter:
    return request.app.state.rate_limiter

@router.post("/api/sessions/{session_id}/messages")
async def post_message(
    session_id: str,
    body: ComposeRequest,
    user: User = Depends(get_current_user),
    rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),
    ...
):
    await rate_limiter.check(user.id)
    ...
```

**429 response body:** `{"error_type": "rate_limited", "detail": "Rate limit exceeded. Try again in N seconds.", "retry_after": N}`. The `Retry-After` HTTP
header is also set.

**No persistence:** The counter lives in memory. On server restart, all counters
reset. This is acceptable for MVP (2-5 users). Rate limiting is per-process: if
the server runs with multiple uvicorn workers, each worker has its own counter and
the effective limit is `N * limit` across the cluster. Multi-worker deployments
need Redis or an equivalent shared store for accurate cross-process rate limiting.
This limitation must be documented in the `WebSettings` docstring. The
`ComposerRateLimiter` interface does not change when moving to Redis.

---

## Finding 4: Edge Insertion Order (elspeth-519a32221d)

### Root Cause

`with_edge()` uses a filter-then-append pattern:

```python
def with_edge(self, edge: EdgeSpec) -> CompositionState:
    edges = tuple(e for e in self.edges if e.id != edge.id) + (edge,)
    return replace(self, edges=edges, version=self.version + 1)
```

When updating an existing edge (same ID), this removes the old edge from its
original position and appends the new one at the end. The order changes on every
update.

`with_node()` uses a different pattern: it finds the index of the existing node
and replaces in-place, preserving position. `with_edge()` should do the same.

### Fix

```python
def with_edge(self, edge: EdgeSpec) -> CompositionState:
    """Add or replace an edge (matched by id). Version incremented."""
    existing_ids = [e.id for e in self.edges]
    if edge.id in existing_ids:
        idx = existing_ids.index(edge.id)
        edge_list = list(self.edges)
        edge_list[idx] = edge
        edges = tuple(edge_list)
    else:
        edges = self.edges + (edge,)
    return replace(self, edges=edges, version=self.version + 1)
```

This is identical to the `with_node()` pattern. The same fix should be applied to
`with_output()` if it has the same problem (check during implementation).

### Impact on Downstream

**YAML generator:** `generate_yaml()` calls `state.to_dict()` which preserves
tuple order as list order. With this fix, edge order in YAML output is stable
across updates. Without it, updating an edge reorders the YAML edges section,
which breaks the determinism guarantee (AC #9 of Sub-Plan 4).

**SpecView rendering (Sub-6):** The frontend renders edges in the order they
appear in the API response. Stable order means the graph does not visually
rearrange when the user updates an edge.

**Test:** Add `test_with_edge_preserves_order` to `test_state.py`, matching the
existing `test_with_node_preserves_order`.

---

## Finding 5: Tool Registry Pattern (elspeth-0a1dae9f90)

### Current State

`execute_tool()` in `tools.py` is a 50-line if/elif chain dispatching 12 tool
names to handler functions. Each handler is already a separate function
(`_execute_set_source`, `_execute_upsert_node`, etc.). The dispatcher is the
only code that needs refactoring.

### Target Pattern

Replace the if-chain with a typed registry dict at module level.

```python
ToolHandler = Callable[
    [dict[str, Any], CompositionState, CatalogServiceProtocol],
    ToolResult,
]

_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    "list_sources": _handle_list_sources,
    "list_transforms": _handle_list_transforms,
    "list_sinks": _handle_list_sinks,
    "get_plugin_schema": _handle_get_plugin_schema,
    "get_expression_grammar": _handle_get_expression_grammar,
    "get_current_state": _handle_get_current_state,
}

# Only these discovery tools are safe to cache. get_current_state returns
# live state that changes with every mutation -- caching it would return
# stale snapshots. Budget classification still uses _DISCOVERY_TOOLS
# (get_current_state IS a discovery turn for budget purposes).
_CACHEABLE_DISCOVERY_TOOLS: frozenset[str] = frozenset({
    "list_sources",
    "list_transforms",
    "list_sinks",
    "get_plugin_schema",
    "get_expression_grammar",
})

_MUTATION_TOOLS: dict[str, ToolHandler] = {
    "set_source": _execute_set_source,
    "upsert_node": _execute_upsert_node,
    "upsert_edge": _execute_upsert_edge,
    "remove_node": _execute_remove_node,
    "remove_edge": _execute_remove_edge,
    "set_metadata": _execute_set_metadata,
}

def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
) -> ToolResult:
    handler = _DISCOVERY_TOOLS.get(tool_name) or _MUTATION_TOOLS.get(tool_name)
    if handler is None:
        return _failure_result(state, f"Unknown tool: {tool_name}")
    return handler(arguments, state, catalog)

def is_cacheable_discovery_tool(name: str) -> bool:
    return name in _CACHEABLE_DISCOVERY_TOOLS
```

**Why two registries plus a cacheable set:** The discovery/mutation split matters
for Finding 1's dual-counter budget. The compose loop needs to know whether a tool
call is discovery or mutation to decide which counter to increment. Exposing
`is_discovery_tool(name: str) -> bool` is trivial with separate registries. The
`_CACHEABLE_DISCOVERY_TOOLS` frozenset is the subset safe for caching -- the cache
check uses this, not `_DISCOVERY_TOOLS`, because `get_current_state` must never be
cached.

### Migration Approach

The handler functions already exist. The refactoring is mechanical:

1. Normalize handler signatures. The current discovery handlers have inconsistent
   signatures (some take `catalog` directly, some use closures). Normalize all
   handlers to accept `(arguments, state, catalog) -> ToolResult`.
2. Build the two registry dicts.
3. Replace the if-chain in `execute_tool()` with a registry lookup.
4. Add `is_discovery_tool(name: str) -> bool` for Finding 1's budget tracking.
5. All existing tests pass unchanged -- `execute_tool()` is the public API and its
   behaviour does not change.

---

## File Map

| Action | Path | Finding(s) |
|--------|------|------------|
| Modify | `src/elspeth/web/composer/state.py` | F4: `with_edge()` insertion order fix |
| Modify | `src/elspeth/web/composer/tools.py` | F5: tool registry refactor, F1: `is_discovery_tool()` + `_CACHEABLE_DISCOVERY_TOOLS` frozenset |
| Modify | `src/elspeth/web/composer/service.py` | F1: dual-counter loop + local discovery cache + `asyncio.wait_for()` timeout, F2: partial state on convergence |
| Modify | `src/elspeth/web/composer/protocol.py` | F1+F2: `ComposerConvergenceError` gains `partial_state` and `budget_exhausted` |
| Create | `src/elspeth/web/middleware/__init__.py` | F3: module init |
| Create | `src/elspeth/web/middleware/rate_limit.py` | F3: `ComposerRateLimiter` |
| Modify | `src/elspeth/web/sessions/routes.py` | F2: partial state persistence on convergence, F3: rate limiter wiring |
| Modify | `src/elspeth/web/app.py` (or equivalent app factory) | F3: `ComposerRateLimiter` instantiation in lifespan |
| Modify | `src/elspeth/web/settings.py` (WebSettings) | F1: replace `composer_max_turns` with dual settings + `composer_timeout_seconds` (85.0) |
| Modify | `tests/unit/web/composer/test_state.py` | F4: `test_with_edge_preserves_order` |
| Modify | `tests/unit/web/composer/test_tools.py` | F5: verify registry dispatch matches if-chain behaviour |
| Modify | `tests/unit/web/composer/test_service.py` | F1: dual-counter tests, F2: partial state on convergence |
| Create | `tests/unit/web/middleware/__init__.py` | F3: test package init |
| Create | `tests/unit/web/middleware/test_rate_limit.py` | F3: rate limiter tests |
| Modify | `docs/superpowers/specs/2026-03-28-web-ux-seam-contracts.md` | F2: update Seam B error shape |

---

## Dependency Analysis

### Blast Radius

All changes are within `src/elspeth/web/`. No core engine code is modified. No
changes to Sub-Plans 1-3 are required.

| Module | Impact |
|--------|--------|
| `composer/state.py` | F4 changes `with_edge()` only. All existing callers (tools, tests, service) are unaffected -- the method signature does not change, only the ordering behaviour. |
| `composer/tools.py` | F5 changes the internal dispatcher structure. `execute_tool()` signature and return types are unchanged. All test callsites are unaffected. |
| `composer/service.py` | F1 and F2 change the compose loop internals. The `compose()` signature and return type are unchanged. The exception type gains fields (breaking change to exception consumers in the route handler). |
| `composer/protocol.py` | F1+F2 add fields to `ComposerConvergenceError`. The route handler in `sessions/routes.py` must be updated to handle the new fields. |
| `middleware/rate_limit.py` | New module. No existing code is affected. The route handler gains a dependency injection call. |
| `sessions/routes.py` | F2 and F3 add new logic to the convergence error handler and the message endpoint respectively. These are additive changes within the existing route function. |
| `settings.py` (WebSettings) | F1 replaces one field with two budget fields and adds `composer_timeout_seconds` (default 85.0). Any code reading `composer_max_turns` breaks at import time (no silent failures). Only `ComposerServiceImpl.__init__` reads these fields. The 85.0s default is intentionally less than the frontend's 90s timeout -- see timeout ordering note in the WebSettings section. Multi-worker deployments need Redis for accurate cross-process rate limiting (documented in the WebSettings docstring). |

### Seam Contract Updates

**Seam B (Composer <-> Session Route):** The 422 error shape gains `budget_exhausted: str`
and optional `partial_state: CompositionStateResponse | null`. This is a
non-breaking additive change to the JSON response (new fields, no removed fields).
The frontend already handles 422 responses; it needs to be updated to check for
`partial_state` but will not break if the field is absent.

---

## Acceptance Criteria

1. Cacheable discovery tool results (list_sources, list_transforms, list_sinks,
   get_plugin_schema, get_expression_grammar) are cached per-compose-call in a
   local `dict` variable, not an instance field. `get_current_state` is explicitly
   excluded from caching (returns live mutable state). Repeated cacheable calls
   with the same arguments return the cached result without incrementing any turn
   counter.

2. The compose loop tracks composition turns and discovery turns separately.
   Budget exhaustion is checked at classification time after each turn -- the loop
   exits only when the specific budget being charged has no remaining capacity
   (not a conjunctive `while A < max AND B < max` guard).
   `ComposerConvergenceError` reports which budget was exhausted.

3. `WebSettings` exposes `composer_max_composition_turns` (default 15),
   `composer_max_discovery_turns` (default 10), and `composer_timeout_seconds`
   (default 85.0, enforced via `asyncio.wait_for()`). The 85.0s default is
   strictly less than the frontend's 90s timeout. The old `composer_max_turns`
   field is removed entirely.

4. On convergence failure, if any mutations were applied, the last valid
   `CompositionState` is attached to `ComposerConvergenceError.partial_state`
   and persisted by the route handler before returning the 422 response. If
   `partial_state.validate()` raises, the state is persisted with
   `is_valid=False` rather than being lost. If `save_composition_state()` fails,
   the error is logged and the 422 response omits `partial_state` (no crash to
   500).

5. The 422 convergence error response body includes `budget_exhausted` (str, one
   of "composition", "discovery", or "timeout") and optionally `partial_state`
   (CompositionStateResponse dict or null).

6. `POST /api/sessions/{id}/messages` enforces per-user rate limiting using
   `WebSettings.composer_rate_limit_per_minute`. The rate limiter is wired as a
   FastAPI `Depends()` dependency, matching the existing `Depends(get_current_user)`
   pattern. Returns HTTP 429 with `Retry-After` header when the limit is exceeded.

7. `CompositionState.with_edge()` preserves insertion order on update (replace
   in-place), consistent with `with_node()`. A test verifies this property.

8. `execute_tool()` dispatches via a registry dict, not an if-chain. Two
   registries (`_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`) plus a
   `_CACHEABLE_DISCOVERY_TOOLS` frozenset support both `is_discovery_tool()` and
   `is_cacheable_discovery_tool()` queries.

9. `ComposerConvergenceError` calls `freeze_fields(self, "partial_state")` in
   `__post_init__` when `partial_state` is not `None`, per the frozen dataclass
   immutability contract.

10. `ComposerRateLimiter` uses per-user `asyncio.Lock` instances (not a single
    global lock). A `_locks_lock` held for microseconds serializes lock
    creation/fetch. The docstring correctly describes asyncio.Lock as guarding
    coroutine suspension points, not thread safety.

11. All existing Sub-Plan 4 tests continue to pass. New tests cover: dual-counter
    convergence (including budget-at-classification-time semantics), partial state
    preservation (including validate failure and persistence failure paths), rate
    limiter allow/deny, edge order preservation, timeout enforcement, and registry
    dispatch for all 12 tools.

12. `with_output()` is audited for the same insertion-order bug as `with_edge()`.
    If the bug exists, it is fixed with the same pattern.

13. `WebSettings` docstring documents that rate limiting is per-process;
    multi-worker deployments require Redis or equivalent shared store for
    cross-process accuracy.
