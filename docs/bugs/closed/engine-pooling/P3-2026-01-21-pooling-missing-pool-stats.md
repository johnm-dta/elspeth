# Bug Report: Pool stats missing required concurrency and delay fields

## Summary

- `PooledExecutor.get_stats()` omits `max_concurrent_reached` and `dispatch_delay_at_completion_ms`, which are specified for node state context in the pooled LLM design. The audit context cannot report peak concurrency or the delay in effect at completion.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/plugins/pooling for bugs; create bug reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Instantiate `PooledExecutor` and call `get_stats()`.
2. Inspect the returned `pool_stats` and `pool_config` dictionaries.

## Expected Behavior

- Stats include `max_concurrent_reached` and `dispatch_delay_at_completion_ms` as specified in the pooling design for audit context.

## Actual Behavior

- Only capacity/success counters and current/peak delay are returned; concurrency and dispatch-at-completion metrics are missing.

## Evidence

- Code: `src/elspeth/plugins/pooling/executor.py:122-140` returns limited stats.
- Spec: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:152-165` lists required fields.

## Impact

- User-facing impact: Reduced observability for pooled execution behavior.
- Data integrity / security impact: Audit context lacks required concurrency and delay metadata.
- Performance or cost impact: None directly.

## Root Cause Hypothesis

- PooledExecutor does not track active concurrency or last dispatch delay at completion.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/plugins/pooling/executor.py`
- Config or schema changes: Add fields in stats output and ensure recorder persists them.
- Tests to add/update: Add tests verifying `max_concurrent_reached` and `dispatch_delay_at_completion_ms` are present.
- Risks or migration steps: None (additive metadata).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:152-165`
- Observed divergence: Required pool stats fields are missing from `get_stats()` output.
- Reason (if known): Not implemented when pooling code moved to `plugins/pooling`.
- Alignment plan or decision needed: Implement concurrency counters and delay snapshotting.

## Acceptance Criteria

- `get_stats()` includes `max_concurrent_reached` and `dispatch_delay_at_completion_ms`.
- Tests demonstrate the metrics are populated under pooled execution.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_pooled_executor.py -k stats`
- New tests required: Yes (stats fields).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md`

## Resolution (2026-02-02)

**Status: FIXED**

### Fix Applied

Added the missing `max_concurrent_reached` and `dispatch_delay_at_completion_ms` fields to `PooledExecutor.get_stats()`.

### Changes Made

1. **`src/elspeth/plugins/pooling/executor.py`**:
   - Added thread-safe concurrency tracking:
     - `_stats_lock`, `_active_workers`, `_max_concurrent`, `_dispatch_delay_at_completion_ms` instance variables
     - `_increment_active_workers()` - increments counter and updates peak
     - `_decrement_active_workers()` - decrements counter
     - `_reset_batch_stats()` - resets per-batch stats at batch start
     - `_capture_completion_stats()` - captures delay at batch end
   - Updated `_execute_single()` to call increment/decrement around semaphore acquire/release
   - Updated `_execute_batch_locked()` to call reset at start and capture at end
   - Updated `get_stats()` to include new fields:
     - `pool_stats.max_concurrent_reached`: Peak concurrent workers during batch
     - `pool_config.dispatch_delay_at_completion_ms`: Throttle delay at batch completion

2. **Tests added** (`tests/plugins/llm/test_pooled_executor.py`):
   - `test_max_concurrent_reached_tracks_peak_workers`
   - `test_dispatch_delay_at_completion_captures_final_delay`
   - `test_stats_reset_between_batches`

### Design Notes

- `max_concurrent_reached` resets per batch (so each batch reports its own peak)
- `dispatch_delay_at_completion_ms` captures the AIMD throttle delay at the exact moment the batch completes
- Thread-safe via `_stats_lock` protecting all stat updates

### Remaining Work

The stats are now **available** via `get_stats()`. Integration with the Landscape recorder (to persist in `context_after_json`) is a separate enhancement that can be done when needed.

---

## Verification (2026-02-01)

**Status: STILL VALID** (at time of verification)

- `get_stats()` still omits `max_concurrent_reached` and `dispatch_delay_at_completion_ms`. (`src/elspeth/plugins/pooling/executor.py:132-150`)

## Verification (2026-01-25)

**Status: STILL VALID**

### Current State

The bug is confirmed valid against current codebase (commit 7540e57 on branch `fix/rc1-bug-burndown-session-4`):

1. **Design specification exists**: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:152-168` specifies that `context_after_json` should include:
   - `pool_stats.max_concurrent_reached`: Peak concurrent requests active
   - `pool_config.dispatch_delay_at_completion_ms`: Throttle delay at batch completion

2. **Implementation omits these fields**: `src/elspeth/plugins/pooling/executor.py:122-141` implements `get_stats()` but only returns:
   - `pool_stats`: `capacity_retries`, `successes`, `peak_delay_ms`, `current_delay_ms`, `total_throttle_time_ms`
   - `pool_config`: `pool_size`, `max_capacity_retry_seconds`

3. **Missing fields never implemented**: The implementation plan (`docs/plans/completed/2026-01-20-pooled-llm-queries-impl.md:1573-1592`) matches the current code exactly - these fields were never carried forward from the design doc to the implementation.

4. **Metrics not collected**: The executor doesn't currently track:
   - Active concurrency count (only has a `Semaphore` for limiting, no counter for peak tracking)
   - Dispatch delay at completion time (only `current_delay_ms` is available, but not captured at specific completion moment)

5. **Stats never actually used**: `get_stats()` is never called in production code - only appears in docstring example at line 76. No LLM transforms or other plugins currently invoke `get_stats()` to populate `context_after_json`.

### Impact Assessment

**Severity remains P3 (minor)**:
- Design-implementation gap: Fields specified in design but never implemented
- No actual impact yet because `get_stats()` isn't integrated into the audit trail recording
- Would need both: (a) implement the missing metrics AND (b) wire up `get_stats()` to `context_after_json` in node state recording
- No evidence this is blocking any current functionality or audit requirements

### Technical Details

**To implement `max_concurrent_reached`**:
- Add thread-safe counter to track active workers (increment when semaphore acquired, decrement when released)
- Track peak value across batch execution
- Return in `pool_stats`

**To implement `dispatch_delay_at_completion_ms`**:
- Snapshot `self._throttle.current_delay_ms` after final result collected in `execute_batch()`
- Return in `pool_config` (though the design doc shows it in pool_config, it's really a runtime stat)

**To actually use these stats**:
- LLM transforms would need to call `self._executor.get_stats()` after batch completion
- Pass result to recorder when creating node state (likely in `context_after` parameter)
- This integration doesn't exist yet

### Git History

No commits found addressing these specific metrics:
- `git log --all --grep="max_concurrent|dispatch_delay_at_completion"` found no fixes
- Pooling code created in commit `0b1cf47` (refactor) and `c786410` (RC-1), neither implemented these fields
- No open plans or TODOs reference implementing these metrics

### Recommendation

**Keep as P3 - enhance if/when audit trail integration is implemented**:

This is a minor design-implementation gap. The missing metrics would provide useful observability, but there's no evidence they're needed for current audit requirements. If pool stats integration with `context_after_json` is implemented in the future, these fields should be added at that time as part of the larger observability enhancement.
