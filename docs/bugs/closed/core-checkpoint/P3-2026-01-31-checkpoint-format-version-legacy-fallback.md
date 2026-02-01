# Bug Report: Legacy compatibility fallback for format_version=None

## Summary

- Checkpoint compatibility check has a date-based fallback for `format_version=None` which violates the No Legacy Code Policy.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/checkpoint/manager.py:234-249` - date cutoff for legacy checkpoints
- Comment: "Legacy checkpoint (format_version is None) - fall back to date check"
- CLAUDE.md:797-841 forbids backwards compatibility code

## Proposed Fix

- Remove date-based fallback; require format_version

## Acceptance Criteria

- Checkpoints without format_version are rejected, not date-checked

## Verification (2026-02-01)

**Status: STILL VALID**

- Compatibility check still falls back to a date-based path when `format_version` is None. (`src/elspeth/core/checkpoint/manager.py:234-249`)

## Closure Report (2026-02-01)

**Status:** CLOSED (IMPLEMENTED)

### Fix Summary

- Removed legacy date-based fallback; unversioned checkpoints are now rejected.

### Test Coverage

- `tests/core/checkpoint/test_manager.py::TestCheckpointManager::test_old_checkpoint_rejected`
