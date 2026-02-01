# Bug Report: PluginContext.record_call bypasses centralized call_index allocation

## Summary

- `PluginContext.record_call()` uses its own `_call_index` counter instead of delegating to `LandscapeRecorder.allocate_call_index()`. If a transform mixes `ctx.record_call()` with audited clients, duplicate `(state_id, call_index)` values can occur.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/context.py:284-295` - `record_call()` uses a local `_call_index` counter when `state_id` is set, bypassing the recorder allocator.
- `src/elspeth/core/landscape/recorder.py:1800-1821` - `allocate_call_index()` is documented as the single source of truth for `(state_id, call_index)` uniqueness.
- Recorder docstring: "All AuditedClient instances MUST delegate to this method rather than maintaining their own counters"

## Impact

- User-facing impact: IntegrityError on calls table if mixing record_call with audited clients
- Data integrity / security impact: Duplicate call_index values corrupt audit trail
- Performance or cost impact: None

## Root Cause Hypothesis

- `PluginContext` maintains its own counter instead of using the centralized allocator, violating the documented contract.

## Proposed Fix

- Code changes:
  - Change `record_call()` to use `self._recorder.allocate_call_index(state_id)` instead of local counter
  - Remove `_call_index` attribute from PluginContext
- Tests to add/update:
  - Add test that interleaves `ctx.record_call()` with audited client calls, verify no duplicate indices

## Acceptance Criteria

- All call_index values come from centralized allocator
- No duplicate `(state_id, call_index)` pairs possible

## Verification (2026-02-01)

**Status: STILL VALID**

- `PluginContext.record_call()` still increments its own `_call_index` instead of calling `allocate_call_index()`. (`src/elspeth/plugins/context.py:284-295`, `src/elspeth/core/landscape/recorder.py:1800-1821`)

---

## RESOLUTION: 2026-02-02

**Status:** FIXED

**Fixed by:** Claude Code (systematic debugging session)

**Implementation:**

Delegated call_index allocation in `PluginContext.record_call()` to the centralized `LandscapeRecorder.allocate_call_index()` method, ensuring UNIQUE(state_id, call_index) when mixing `ctx.record_call()` with audited clients.

### Files Changed

1. **`src/elspeth/plugins/context.py`**
   - Removed local `_call_index` counter and `_call_index_lock` fields
   - Changed `record_call()` to call `self.landscape.allocate_call_index(state_id)` instead of incrementing local counter
   - Removed `__post_init__` method (only existed to initialize the lock)
   - Removed unused `Lock` import

2. **`tests/core/landscape/test_operations.py`**
   - Added `test_plugin_context_record_call_uses_centralized_allocator()` that verifies interleaved allocator calls and `ctx.record_call()` produce unique indices

### Code Evidence

**Before (local counter - BUGGY):**
```python
# In PluginContext
_call_index: int = field(default=0)
_call_index_lock: Lock = field(init=False)

def record_call(self, ...):
    if has_state:
        with self._call_index_lock:
            call_index = self._call_index  # ❌ Local counter
            self._call_index += 1
        recorded_call = self.landscape.record_call(
            state_id=self.state_id,
            call_index=call_index,  # ❌ Collides with audited clients
            ...
        )
```

**After (centralized allocator - FIXED):**
```python
# In PluginContext
# Note: call_index allocation is delegated to LandscapeRecorder.allocate_call_index()

def record_call(self, ...):
    if has_state:
        call_index = self.landscape.allocate_call_index(self.state_id)  # ✅ Centralized
        recorded_call = self.landscape.record_call(
            state_id=self.state_id,
            call_index=call_index,  # ✅ Coordinates with audited clients
            ...
        )
```

### Why This Fix Works

**Cross-mechanism coordination:**
```python
# Scenario: Transform uses both audited client AND ctx.record_call()
response = await ctx.llm_client.query(...)  # recorder.allocate_call_index() → 0
ctx.record_call(...)                         # recorder.allocate_call_index() → 1 ✅

# Both use the same allocator for the same state_id
# No collision on (state_id, call_index) ✅
```

**Consistency with P1-2026-01-21 fix:**
This fix completes the work started in P1-2026-01-21-call-index-collisions-across-clients. That fix centralized allocation for `AuditedClientBase`, but `PluginContext.record_call()` was not updated to use the same allocator. Now all call index allocation goes through `LandscapeRecorder.allocate_call_index()`.

### Test Evidence

```
tests/core/landscape/test_operations.py::TestCallIndexUniquenessConstraints::test_plugin_context_record_call_uses_centralized_allocator PASSED
```

All 41 tests in `test_operations.py` pass, including the new test that verifies interleaved allocator calls and `ctx.record_call()` produce unique indices.
