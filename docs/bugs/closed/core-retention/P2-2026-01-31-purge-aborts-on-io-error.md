# Bug Report: Purge aborts on payload-store I/O errors, skipping grade updates

## Summary

- Calls to `payload_store.exists()` and `payload_store.delete()` are not wrapped in try/except. An OSError or PermissionError aborts the purge loop.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/retention/purge.py:393-401` - no try/except around `exists()` / `delete()` I/O operations.
- Code has `failed_refs` list but exceptions abort before populating it

## Impact

- User-facing impact: Partial purge with inconsistent grade updates
- Data integrity: Some refs purged, others not, grade not updated

## Proposed Fix

- Wrap I/O operations in try/except, collect failures, update grade for successful deletions

## Acceptance Criteria

- I/O errors don't abort entire purge
- Grade updates reflect actual state

## Resolution (2026-02-02)

**Status: FIXED**

### Root Cause
The `purge_payloads()` method iterated through refs calling `exists()` and `delete()` without try/except blocks. Any `OSError` or `PermissionError` would propagate and abort the entire loop, leaving the system in an inconsistent state where some refs were deleted but grade updates were never executed.

### Fix Applied
Wrapped both `exists()` and `delete()` calls in try/except blocks that catch `OSError` (parent of `PermissionError`, `IOError`, etc.). On exception, the ref is added to `failed_refs` and the loop continues with remaining refs. Grade updates now correctly execute for successful deletions even when some refs fail.

**Files Modified:**
- `src/elspeth/core/retention/purge.py:393-416` - Added try/except around I/O operations

**Tests Added:**
- `tests/core/retention/test_purge.py::TestPurgeIOErrorHandling` - 3 new tests verifying:
  - Purge continues after `exists()` raises exception
  - Purge continues after `delete()` raises exception
  - Grade updates happen for successful deletions even when some fail
