# Bug Report: can_resume propagates IncompatibleCheckpointError instead of returning ResumeCheck

## Summary

- `can_resume()` doesn't wrap `get_latest_checkpoint()` in try/except, so `IncompatibleCheckpointError` propagates instead of returning a `ResumeCheck` explaining why resume isn't possible.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/checkpoint/recovery.py:93-99` - call to `get_latest_checkpoint()` not wrapped.
- `src/elspeth/core/checkpoint/manager.py:122-160` - `get_latest_checkpoint()` can raise `IncompatibleCheckpointError`.
- `can_resume()` contract is to return `ResumeCheck`, not raise

## Impact

- User-facing impact: Exception instead of helpful error message
- Data integrity: None

## Proposed Fix

- Catch `IncompatibleCheckpointError` and return `ResumeCheck(can_resume=False, reason=...)`

## Acceptance Criteria

- Incompatible checkpoints result in ResumeCheck with explanation, not exception

## Verification (2026-02-01)

**Status: STILL VALID**

- `can_resume()` still calls `get_latest_checkpoint()` without catching `IncompatibleCheckpointError`. (`src/elspeth/core/checkpoint/recovery.py:93-99`, `src/elspeth/core/checkpoint/manager.py:122-160`)

## Resolution (2026-02-02)

**Status: FIXED**

**Fix:** Added try/except block in `RecoveryManager.can_resume()` to catch `IncompatibleCheckpointError` and return `ResumeCheck(can_resume=False, reason=str(e))` instead of propagating the exception.

**Test added:**
- `test_can_resume_returns_check_for_incompatible_checkpoint` - verifies can_resume() returns ResumeCheck for incompatible checkpoints, maintaining API contract
