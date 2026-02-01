# Bug Report: CallVerifier misclassifies error calls as missing payloads

## Summary

- `CallVerifier` treats `recorded_response is None` as `payload_missing=True`, but ERROR calls with no response_ref are legitimate (call failed before response).

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/clients/verifier.py:208-222` - treats `recorded_response is None` as missing without checking call status.
- Never checks `call.status` or `call.error_json`
- ERROR calls with no response_ref are legitimate

## Impact

- User-facing impact: False "missing payload" reports for error calls
- Data integrity: None (verification, not recording)

## Proposed Fix

- Check call.status before marking as payload_missing; ERROR calls with no response are expected

## Acceptance Criteria

- ERROR calls with no response_ref not counted as missing payloads

## Verification (2026-02-01)

**Status: STILL VALID**

- `CallVerifier` still marks `payload_missing` when `recorded_response` is `None` without checking `call.status`. (`src/elspeth/plugins/clients/verifier.py:208-222`)

## Resolution (2026-02-02)

**Status: FIXED**

### Root Cause

The code checked only `recorded_response is None` to determine `payload_missing`. This conflated two distinct cases:
1. Call never had a response (`response_ref=None`) - legitimate for connection failures
2. Call had a response but it was purged (`response_ref` set, `response=None`) - actual missing payload

### Fix Applied

Changed `verifier.py:211-222` to check `call.response_ref`:

```python
# Before (buggy):
if recorded_response is None:
    # ... mark as payload_missing

# After (fixed):
if recorded_response is None and call.response_ref is not None:
    # ... mark as payload_missing (response was recorded but purged)

if recorded_response is None:
    # Call never had a response - return result but NOT payload_missing
```

### Tests Added/Updated

- `test_verify_with_purged_response_payload` - Purged payloads correctly flagged
- `test_verify_error_call_without_response_not_missing_payload` - Error calls without response not flagged
- `test_verify_error_call_with_purged_response_is_missing_payload` - Error calls with purged response flagged

### Verification

- All 34 verifier tests pass
- No regressions introduced
