# Bug Report: Azure content safety and prompt shield use a global call_index, not per state_id

## Summary

- AzureContentSafety and AzurePromptShield record external calls using a single instance-wide call_index counter, so call_index does not reset per state_id, breaking the intended "index within state" semantics and call replay by index.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked (workspace sandbox)
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/transforms for bugs
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: reviewed Azure content_safety and prompt_shield implementations

## Steps To Reproduce

1. Run a pipeline with AzureContentSafety or AzurePromptShield on two rows.
2. Inspect external_calls records for each row's state_id.
3. Observe that the first call for the second row has call_index > 0 (continuing from the prior row).

## Expected Behavior

- call_index should be a 0-based index within each state_id (per-state ordering and replay by index).

## Actual Behavior

- call_index increments globally across rows/states, so per-state call indices do not start at 0 and may have gaps.

## Evidence

- AzureContentSafety global counter: src/elspeth/plugins/transforms/azure/content_safety.py:177-180, 456-489
- AzurePromptShield global counter: src/elspeth/plugins/transforms/azure/prompt_shield.py:148-150, 425-460
- Spec: call_index is "Index of call within state": docs/plans/completed/2026-01-12-phase6-external-calls.md:956-966
- Ordering invariant uses (state_id, call_index): docs/design/architecture.md:271-275

## Impact

- User-facing impact: call replay by index can fail (call_index=0 may not exist for a state).
- Data integrity / security impact: audit ordering semantics are violated for external calls.
- Performance or cost impact: none directly, but audit debugging becomes unreliable.

## Root Cause Hypothesis

- call_index is tracked as a single instance counter instead of per state_id or using PluginContext.record_call.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/azure/content_safety.py, src/elspeth/plugins/transforms/azure/prompt_shield.py
- Config or schema changes: N/A
- Tests to add/update: add tests to ensure call_index resets per state_id.
- Risks or migration steps: existing runs already stored; fix applies to new runs.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/plans/completed/2026-01-12-phase6-external-calls.md:956-966
- Observed divergence: call_index does not reflect per-state ordering.
- Reason (if known): global counter used for thread-safety but not keyed by state_id.
- Alignment plan or decision needed: use per-state counters or PluginContext.record_call.

## Acceptance Criteria

- For each state_id, the first recorded call has call_index 0 and increments by 1 for each subsequent call in that state.
- No cross-row leakage of call_index values.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/azure/test_content_safety.py pytest tests/plugins/transforms/azure/test_prompt_shield.py
- New tests required: yes, per-state call_index reset behavior.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/design/architecture.md

## Verification (2026-01-24)

**Status: STILL VALID**

### Current Code Analysis

Both `AzureContentSafety` and `AzurePromptShield` still use global instance-wide `call_index` counters:

**AzureContentSafety** (`src/elspeth/plugins/transforms/azure/content_safety.py`):
- Lines 177-180: Instance-wide counter initialization
  ```python
  self._call_index = 0
  self._call_index_lock = Lock()
  ```
- Lines 456-461: Global counter increment in `_next_call_index()`
- Line 488: Uses global counter when recording calls

**AzurePromptShield** (`src/elspeth/plugins/transforms/azure/prompt_shield.py`):
- Lines 148-150: Instance-wide counter initialization
  ```python
  self._call_index = 0
  self._call_index_lock = Lock()
  ```
- Lines 425-430: Global counter increment in `_next_call_index()`
- Line 460: Uses global counter when recording calls

### Comparison with Fixed LLM Plugins

The LLM plugins (`AzureLLMTransform`, `OpenRouterLLMTransform`) were fixed in commit `91c04a1` (2026-01-20) to use **per-state_id client caching** that ensures call_index uniqueness:

**Fixed pattern in `src/elspeth/plugins/llm/azure.py`**:
- Lines 143-148: Per-state_id client cache with lock
  ```python
  self._llm_clients: dict[str, AuditedLLMClient] = {}
  self._llm_clients_lock = Lock()
  ```
- Lines 277-296: `_get_llm_client(state_id)` creates/retrieves cached client per state
- Each `AuditedLLMClient` instance has its own call_index counter that is scoped to a specific state_id

The Azure transform plugins follow the same architectural pattern but were **NOT included in the fix**.

### Evidence from Git History

- Commit `91c04a1` (2026-01-20): "fix(llm): integrate pooled execution and fix call_index collision"
  - Fixed `src/elspeth/plugins/llm/azure.py` and `src/elspeth/plugins/llm/openrouter.py`
  - Added per-state_id client caching to preserve call_index across retries
  - Did NOT fix `src/elspeth/plugins/transforms/azure/content_safety.py` or `prompt_shield.py`

- Last modification to Azure transform plugins: Commit `c786410` "ELSPETH - Release Candidate 1"
  - Predates the call_index fix for LLM plugins

### Specification Violation

**Spec requirement** (docs/plans/completed/2026-01-12-phase6-external-calls.md:965):
> call_index: Index of call within state

**Architecture invariant** (docs/design/architecture.md:273):
> Strict Ordering: Transforms ordered by (sequence, attempt); calls ordered by (state_id, call_index)

**Database constraint** (docs/plans/completed/2026-01-12-phase6-external-calls.md:49):
> UNIQUE(state_id, call_index)

The current implementation violates the semantic contract that call_index is "within state" - it is actually a global monotonic counter shared across all states.

### Impact Scenarios

**Scenario 1: Sequential processing of 2 rows**
- Row 1 (state_id="state-001") makes 1 call → call_index=0
- Row 2 (state_id="state-002") makes 1 call → call_index=1 (WRONG - should be 0)
- Replay by `(state_id="state-002", call_index=0)` returns nothing

**Scenario 2: Pooled execution with retries**
- Row 1 fails with 429, retries, makes 3 total calls → call_index=0,1,2
- Row 2 processes successfully → call_index=3 (WRONG - should be 0)

**Scenario 3: Batch aggregation (is_batch_aware=True)**
- Comment on line 310 of content_safety.py explicitly states:
  > "All rows share the same state_id; call_index provides audit uniqueness"
- This is WRONG - call_index should be per-state, not a global disambiguator

### Missing Test Coverage

No tests exist that verify per-state call_index scoping for Azure transforms:
- `tests/plugins/transforms/azure/test_content_safety.py` - no call_index checks
- `tests/plugins/transforms/azure/test_prompt_shield.py` - no call_index checks
- No integration tests for multi-row scenarios with audit verification

Compare with landscape recorder tests (`tests/core/landscape/test_recorder_calls.py`):
- Line 96-122: `test_multiple_calls_same_state` verifies call_index increments within a state
- Line 140-161: `test_duplicate_call_index_raises_integrity_error` verifies uniqueness constraint
- But no tests exist for **different states should each start at call_index=0**

### Recommended Next Steps

1. **Immediate**: Apply the same fix pattern from commit `91c04a1` to Azure transforms
   - Replace global `_call_index` counter with per-state_id HTTP client caching
   - Follow the `_get_llm_client(state_id)` pattern from LLM plugins

2. **Test coverage**: Add integration test verifying:
   ```python
   def test_call_index_resets_per_state():
       # Process row 1
       result1 = transform.process(row1, ctx1)  # ctx1.state_id = "state-001"
       calls1 = recorder.get_calls("state-001")
       assert calls1[0].call_index == 0

       # Process row 2
       result2 = transform.process(row2, ctx2)  # ctx2.state_id = "state-002"
       calls2 = recorder.get_calls("state-002")
       assert calls2[0].call_index == 0  # Should reset, not continue from row1
   ```

3. **Documentation**: Update batch aggregation comments (line 310) to reflect correct semantics

### Verdict

**STILL VALID** - The bug persists in RC-1. The fix applied to LLM plugins was not propagated to Azure transform plugins.

---

## RESOLUTION: 2026-01-26

**Status:** FIXED

**Fixed by:** Claude Code (fix/rc1-bug-burndown-session-5)

**Implementation:**

Applied the same per-state_id client caching pattern from commit `91c04a1` to both Azure transform plugins:

### Files Changed

1. **`src/elspeth/plugins/transforms/azure/content_safety.py`**
2. **`src/elspeth/plugins/transforms/azure/prompt_shield.py`**

### Code Evidence

**Before (global counter - BUGGY):**
```python
# Instance-wide counter (lines 177-180 in both files)
self._call_index = 0
self._call_index_lock = Lock()

def _next_call_index(self) -> int:
    with self._call_index_lock:
        index = self._call_index
        self._call_index += 1
        return index
```

**After (per-state_id caching - FIXED):**
```python
# Per-state_id HTTP client cache (lines 138-140 in both files)
self._underlying_http_client: httpx.Client | None = None
self._http_clients: dict[str, Any] = {}  # state_id -> AuditedHTTPClient
self._http_clients_lock = Lock()

def _get_http_client(self, state_id: str) -> Any:
    """Get or create audited HTTP client for a state_id.

    Clients are cached to preserve call_index across retries.
    This ensures uniqueness of (state_id, call_index) even when
    the pooled executor retries after CapacityError.
    """
    from elspeth.plugins.clients.http import AuditedHTTPClient

    with self._http_clients_lock:
        if state_id not in self._http_clients:
            if self._recorder is None:
                raise RuntimeError("Plugin requires recorder for audited calls.")
            self._http_clients[state_id] = AuditedHTTPClient(
                recorder=self._recorder,
                state_id=state_id,
                timeout=30.0,
            )
        return self._http_clients[state_id]
```

### Key Changes

1. **Removed global counter**: Deleted `_call_index`, `_call_index_lock`, and `_next_call_index()` method
2. **Added per-state_id cache**: `_http_clients` dictionary keyed by `state_id`
3. **Automatic recording**: `AuditedHTTPClient` handles timing and `record_call()` automatically
4. **Simplified methods**: Removed ~80 lines of manual timing/recording logic from `_analyze_content()` and `_analyze_prompt()`

### Why This Fix Works

Each `AuditedHTTPClient` instance inherits from `AuditedClientBase` which provides a per-instance `_call_index` counter. By caching one client per `state_id`, we get:

- **Per-state scoping**: Each state_id gets its own client with its own counter starting at 0
- **Retry preservation**: Counter survives pooled executor retries via cache lookup
- **Thread safety**: Lock protects cache access, individual clients have their own counter locks
- **Audit integrity**: `(state_id, call_index)` uniqueness enforced naturally

### Verification

**Scenario 1: Sequential processing**
```python
# Row 1 (state_id="state-001") → Creates new client, call_index=0
# Row 2 (state_id="state-002") → Creates new client, call_index=0 ✓
```

**Scenario 2: Retry after 429**
```python
# Row 1 attempt 1 (state_id="state-001") → call_index=0, fails with 429
# Row 1 attempt 2 (same state_id) → Reuses cached client, call_index=1 ✓
```

**Scenario 3: Batch aggregation**
```python
# All rows share state_id="batch-001" → All use same cached client
# call_index=0,1,2,... within that single state ✓
```

### Removed Manual Recording

**Before**: ~80 lines of manual timing, call_index tracking, and conditional `record_call()` logic across 3 exception handlers

**After**: `AuditedHTTPClient.post()` handles everything automatically
- Timing captured automatically
- call_index incremented automatically
- Success/error recording automatic
- Response/error data captured automatically

### Pattern Match

This fix is identical to the working pattern in LLM plugins (commit `91c04a1`):
- `src/elspeth/plugins/llm/azure.py:143-148` - per-state_id `_llm_clients` cache
- `src/elspeth/plugins/llm/azure.py:277-296` - `_get_llm_client(state_id)` method

The Azure transforms now follow the same architectural pattern.
