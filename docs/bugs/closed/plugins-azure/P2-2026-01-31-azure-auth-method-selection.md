# Bug Report: Auth method selection ignores validator's trimmed checks

## Summary

- Validator uses `self.connection_string.strip()` but runtime methods use raw truthiness `if self.connection_string:`. Whitespace-only strings pass validator but trigger wrong branch at runtime.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/azure/auth.py:85-99` - validator uses `.strip()`
- Lines 157-167 - `create_blob_service_client()` uses raw `if self.connection_string:`
- Lines 210-217 - `auth_method` property uses same raw truthiness

## Impact

- User-facing impact: Whitespace-only connection_string causes confusing behavior
- Data integrity: None

## Proposed Fix

- Use `.strip()` consistently in all auth method checks

## Acceptance Criteria

- Whitespace-only strings treated as empty consistently

## Verification (2026-02-01)

**Status: FIXED**

- Added `_is_set()` helper method that checks `value is not None and bool(value.strip())`
- Updated `auth_method` property and `create_blob_service_client()` to use `_is_set()`
- Added 3 regression tests in `TestAzureAuthConfigWhitespaceConsistency`
- All 28 tests pass, mypy clean
