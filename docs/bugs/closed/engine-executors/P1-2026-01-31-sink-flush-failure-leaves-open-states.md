# Bug Report: Sink flush failures leave sink node_states OPEN

## Summary

- When `sink.flush()` raises an exception, sink node_states are left in OPEN status instead of being completed with FAILED, violating the terminal state requirement.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- **Original issue (now fixed):** `sink.flush()` previously had no exception handling, so failures left OPEN node_states.
- **Current code (fix in place):** `sink.flush()` is wrapped and completes all node_states as FAILED before re-raising. (`src/elspeth/engine/executors.py:1720-1741`)

## Impact

- User-facing impact: Runs can crash with dangling OPEN node_states
- Data integrity / security impact: Audit trail has incomplete state transitions
- Performance or cost impact: May complicate resume and debugging

## Root Cause Hypothesis

- Exception handling was added for `sink.write()` but not for `sink.flush()`.

## Proposed Fix

- Code changes:
  - Wrap `sink.flush()` in try/except similar to `sink.write()`
  - On flush failure, complete all pending node_states with FAILED status
- Tests to add/update:
  - Add test that mocks `sink.flush()` to raise, verify node_states are completed as FAILED

## Acceptance Criteria

- Sink flush failures result in FAILED node_states, not OPEN
- No dangling OPEN states after sink exceptions

## Verification (2026-02-01)

**Status: FIXED**

- `sink.flush()` exceptions are now handled and all open node_states are completed as FAILED before raising. (`src/elspeth/engine/executors.py:1720-1741`)

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- The fix is present in `SinkExecutor.write()` and preserves audit invariants on flush failure.
- No additional remediation required beyond regression coverage.
