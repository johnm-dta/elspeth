# Bug Report: Call indices collide across audited clients sharing a state

## Summary

- `AuditedClientBase` tracks call indices per client instance; using multiple audited clients (HTTP + LLM) in the same node state yields duplicate `(state_id, call_index)` and triggers DB integrity errors.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/clients` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of audited client base and landscape schema

## Steps To Reproduce

1. Create a transform that instantiates both `AuditedHTTPClient` and `AuditedLLMClient` with the same `state_id`.
2. Make one call with each client in the same node state.
3. Observe the second `record_call` fails with `IntegrityError` on the `calls(state_id, call_index)` unique constraint.

## Expected Behavior

- Call indices are unique per `state_id` across all external calls in a node state, regardless of which client type makes the call.

## Actual Behavior

- Each audited client starts its own counter at 0, so multiple clients in the same state generate duplicate `call_index` values.

## Evidence

- Per-client call index counter: `src/elspeth/plugins/clients/base.py:41-58`
- Unique constraint on `(state_id, call_index)`: `src/elspeth/core/landscape/schema.py:188-204`

## Impact

- User-facing impact: pipelines that use multiple audited clients in a single transform crash when recording calls.
- Data integrity / security impact: calls may be missing from the audit trail if collisions occur.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Call indices are scoped to individual client instances instead of the node state, but the database enforces uniqueness per state.

## Proposed Fix

- Code changes (modules/files):
  - Centralize call index allocation by `state_id` (e.g., in `LandscapeRecorder` or `PluginContext`) and have audited clients request the next index.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that uses two audited clients with the same `state_id` and asserts both calls record successfully with distinct indices.
- Risks or migration steps:
  - Ensure any cached clients preserve monotonic indices across retries.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/core/landscape/schema.py:203` (unique constraint on call indices)
- Observed divergence: audited clients do not coordinate indices across client types.
- Reason (if known): per-client counter was simpler to implement.
- Alignment plan or decision needed: adopt a shared per-state counter.

## Acceptance Criteria

- Multiple audited clients can record calls under the same `state_id` without integrity errors.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k call_index`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Code Inspection

The bug remains unfixed in the current codebase:

1. **AuditedClientBase still uses per-instance counters** (`src/elspeth/plugins/clients/base.py:43-58`):
   - Each client instance initializes `self._call_index = 0` in `__init__`
   - `_next_call_index()` increments its own counter
   - No changes to this file since bug was reported (commit ae2c0e6)

2. **Database constraint unchanged** (`src/elspeth/core/landscape/schema.py:203`):
   - `UniqueConstraint("state_id", "call_index")` still enforces uniqueness per state
   - No mechanism to coordinate indices across multiple client instances

3. **PluginContext has alternative mechanism** (`src/elspeth/plugins/context.py:93, 191-238`):
   - Has its own `_call_index` field and `record_call()` method
   - This provides a shared counter per context, but AuditedClientBase doesn't use it
   - The two mechanisms exist in parallel without coordination

### Current Patterns in Codebase

**Pattern 1: AuditedClientBase with per-instance counters**
- Used by `AuditedHTTPClient` and `AuditedLLMClient`
- Each client creates its own counter starting at 0
- Bug: Two clients with same `state_id` will generate duplicate indices

**Pattern 2: Transform-owned counters**
- `AzurePromptShield` has its own `_call_index` counter (lines 152-153)
- Calls `recorder.record_call()` directly with its own index
- Avoids AuditedClientBase entirely

**Pattern 3: PluginContext.record_call()**
- Context-scoped counter that increments on each call
- Transforms could use this instead of audited clients
- Not currently used by any audited client implementations

### Risk Assessment

**Current risk: LOW** - No production scenarios trigger this bug yet:

1. **No transforms use both client types simultaneously**
   - LLM transforms use `AuditedLLMClient` only
   - HTTP transforms (Azure) use their own counters, not `AuditedHTTPClient`
   - No hybrid transforms exist in the codebase

2. **PluginContext provides both `llm_client` and `http_client` fields** (lines 97-98):
   - Architecture supports multiple clients per state
   - But no current usage actually instantiates both

3. **Future risk is HIGH if:**
   - A transform needs both LLM and HTTP calls (e.g., retrieve docs via HTTP, then query LLM)
   - Multiple LLM providers in same transform (fallback pattern)
   - Hybrid orchestration patterns emerge

### Test Coverage

Examined test files that reference `call_index`:
- `tests/plugins/clients/test_audited_client_base.py`: Tests thread safety within single client, NOT collision across clients
- No tests exist for the scenario: two different client instances sharing same `state_id`

### Recommendation

**Priority: P1 maintained** - While not currently triggered, this is a correctness bug that violates audit integrity guarantees:

1. **Immediate fix**: Centralize call index allocation in `LandscapeRecorder` or have `AuditedClientBase` delegate to `PluginContext._call_index`
2. **Add regression test**: Create two different client types with same `state_id`, verify both record successfully with unique indices
3. **Architectural alignment**: Eliminate Pattern 2 (transform-owned counters) in favor of consistent PluginContext mechanism

The bug report's proposed fix remains valid and should be implemented before any transform attempts to use multiple client types.

---

## RESOLUTION: 2026-01-26

**Status:** FIXED

**Fixed by:** Claude Code (fix/rc1-bug-burndown-session-5)

**Implementation:**

Centralized call_index allocation in `LandscapeRecorder` to ensure UNIQUE(state_id, call_index) across all client types and retry attempts.

### Files Changed

1. **`src/elspeth/core/landscape/recorder.py`**
   - Added `_call_indices: dict[str, int]` and `_call_index_lock` to `__init__`
   - Added `allocate_call_index(state_id)` method for centralized allocation

2. **`src/elspeth/plugins/clients/base.py`**
   - Removed per-instance `_call_index` and `_call_index_lock` fields
   - Refactored `_next_call_index()` to delegate to `recorder.allocate_call_index()`

### Code Evidence

**Before (per-client counters - BUGGY):**
```python
class AuditedClientBase:
    def __init__(self, recorder, state_id):
        self._recorder = recorder
        self._state_id = state_id
        self._call_index = 0  # ❌ Per-client counter
        self._call_index_lock = Lock()

    def _next_call_index(self) -> int:
        with self._call_index_lock:
            idx = self._call_index
            self._call_index += 1
            return idx  # ❌ Independent counters collide
```

**After (centralized allocation - FIXED):**
```python
# In LandscapeRecorder
class LandscapeRecorder:
    def __init__(self, db, *, payload_store=None):
        self._db = db
        self._payload_store = payload_store
        self._call_indices: dict[str, int] = {}  # ✅ Per-state_id counter
        self._call_index_lock = Lock()

    def allocate_call_index(self, state_id: str) -> int:
        """Centralized allocation - single source of truth."""
        with self._call_index_lock:
            if state_id not in self._call_indices:
                self._call_indices[state_id] = 0
            idx = self._call_indices[state_id]
            self._call_indices[state_id] += 1
            return idx  # ✅ Coordinated across all clients

# In AuditedClientBase
class AuditedClientBase:
    def __init__(self, recorder, state_id):
        self._recorder = recorder
        self._state_id = state_id
        # ✅ NO per-instance counter

    def _next_call_index(self) -> int:
        """Delegate to recorder - ensures coordination."""
        return self._recorder.allocate_call_index(self._state_id)
```

### Why This Fix Works

**Scope Alignment:**
- Database constraint: `UNIQUE(state_id, call_index)` - persistent per state_id
- LandscapeRecorder: Per-state_id counter - matches database scope ✅
- Lifecycle: Recorder persists across retries within same run ✅

**Cross-client coordination:**
```python
# Scenario: Transform uses both HTTP and LLM clients
http_client = AuditedHTTPClient(recorder, state_id="state-001")
llm_client = AuditedLLMClient(recorder, state_id="state-001")

http_client.post(...)  # recorder.allocate_call_index("state-001") → 0
llm_client.query(...)   # recorder.allocate_call_index("state-001") → 1 ✅

# Both share the same recorder's counter for state-001
# No collision on (state-001, call_index) ✅
```

**Retry preservation:**
```python
# Attempt 1: state_id="state-001"
client.post(...)  # allocate → 0, fails with 429

# Retry (same state_id, same recorder)
client.post(...)  # allocate → 1 ✅ (counter persisted in recorder)
```

### Benefits

1. **Eliminates per-state_id caching boilerplate** - No more `_get_http_client(state_id)` methods in every plugin
2. **Architectural clarity** - Recorder owns audit state, including call indices
3. **Thread-safe** - Single lock protects centralized state
4. **Future-proof** - Automatically coordinates any new client types (SQL, FileSystem, etc.)

### Impact on Previous Fix

This fix **supersedes and simplifies** the per-state_id client caching pattern we implemented in commit `d93a315` (ContentSafety/PromptShield):

**Old pattern (commit d93a315):**
- Each plugin maintains `_http_clients: dict[str, AuditedHTTPClient]`
- Cache one client per state_id
- Client has per-instance counter
- **Works but requires boilerplate in every plugin**

**New pattern (this commit):**
- Plugins create clients normally (no caching needed)
- All clients delegate to centralized recorder
- **Zero boilerplate, works automatically**

The per-state_id caching can be removed in a future cleanup commit, but it still works correctly with this fix (just unnecessary now).
