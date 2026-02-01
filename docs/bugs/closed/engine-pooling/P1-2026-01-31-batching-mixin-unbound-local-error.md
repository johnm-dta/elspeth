# Bug Report: Release loop exception path uses uninitialized token/state_id

## Summary

- In `BatchAwareMixin`, if `wait_for_next_release()` raises before the unpacking line, the exception handler references `token` and `state_id` which are never assigned, causing `UnboundLocalError`.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/batching/mixin.py:262-309`:
  - Line 268: `token, result, state_id = entry.result` unpacks after successful wait
  - Exception handler at lines 287-309 references `token` and `state_id`
  - If `wait_for_next_release()` raises before line 268 (e.g., invariant violation at line 254-257), variables are never assigned
  - Line 299 attempts to emit with `token`, causing `UnboundLocalError`

## Impact

- User-facing impact: Secondary exception masks original error, confusing debugging
- Data integrity / security impact: None (crashes)
- Performance or cost impact: Pipeline hangs on timeout waiting for error handling

## Root Cause Hypothesis

- Exception handler assumes `token`/`state_id` are always assigned, but they're only assigned after successful `wait_for_next_release()`.

## Proposed Fix

- Code changes:
  - Initialize `token = None`, `state_id = None` before try block
  - Check for None in exception handler before using
  - Or: Restructure to only reference variables that are guaranteed to be assigned
- Tests to add/update:
  - Add test that forces `wait_for_next_release()` to raise, verify no UnboundLocalError

## Acceptance Criteria

- Exception handler works correctly even when `wait_for_next_release()` fails
- Original error is properly surfaced, not masked by UnboundLocalError
