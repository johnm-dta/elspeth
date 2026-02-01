# Bug Report: Replay ignores missing payloads for error calls with recorded responses

## Summary

- Replayer uses `response_data = {}` fallback for error calls with missing payloads, but ERROR calls can have `response_ref` set (HTTP errors return bodies). If purged, replay substitutes empty dict.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/clients/replayer.py:201-220`:
  - Lines 212-216: Missing payload check only fails if `not was_error`.
  - Lines 218-220: For error calls, `response_data = {}` used as fallback.
- Silent data loss violates audit principles

## Impact

- User-facing impact: Replay produces different results if error response purged
- Data integrity: Replay not faithful to original

## Proposed Fix

- Fail replay if response_ref exists but payload missing, regardless of error status

## Acceptance Criteria

- Missing payloads always cause replay failure, not silent substitution

## Verification (2026-02-01)

**Status: STILL VALID**

- Replayer still substitutes `{}` for missing error responses instead of failing. (`src/elspeth/plugins/clients/replayer.py:212-220`)

## Resolution (2026-02-02)

**Status: FIXED**

### Root Cause

The code checked `call.status == CallStatus.ERROR` to decide whether missing payload was acceptable. This was wrong because ERROR calls can legitimately have response bodies (HTTP 400/500 errors with bodies).

### Key Insight

The correct distinction is `response_ref`:
- `response_ref=None` → call never had a response (OK to use `{}` for errors)
- `response_ref` set but `response_data=None` → response was purged → FAIL

### Fix Applied

Changed `replayer.py:212-220` from checking `not was_error` to checking `response_ref`:

```python
# Before (buggy):
if response_data is None and not was_error:
    raise ReplayPayloadMissingError(...)
if response_data is None:
    response_data = {}

# After (fixed):
if response_data is None and call.response_ref is not None:
    raise ReplayPayloadMissingError(...)
if response_data is None:
    response_data = {}
```

### Tests Added/Updated

- `test_replay_handles_error_calls_without_response` - Error calls with no response_ref get `{}`
- `test_replay_error_call_with_purged_response_raises` - Error calls with purged response fail
- `test_replay_error_call_with_response_succeeds` - Error calls with available response work
- Updated `test_replay_with_purged_response_data` to set `response_ref` (more realistic)

### Verification

- All 17 replayer tests pass
- No regressions introduced
