# Bug Report: Retention purge excludes failed runs, leaving their payloads undeletable

## Summary

- Purge uses `runs_table.c.status == "completed"` for expired condition, so failed runs are treated as "active" regardless of age.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/retention/purge.py:94-96` and `137-141` - uses `status == "completed"` to define expired runs.
- Failed runs older than retention cutoff are excluded from purge

## Impact

- User-facing impact: Unbounded storage growth for failed runs
- Data integrity: None

## Proposed Fix

- Include failed runs (status != "running") in purge eligibility

## Acceptance Criteria

- Failed runs older than retention period are eligible for payload purge

## Resolution (2026-02-02)

**Status: FIXED**

### Root Cause
The `run_expired_condition` used `status == "completed"`, which excluded failed runs. The `run_active_condition` used `status != "completed"`, which incorrectly treated failed runs as "active" regardless of age.

### Fix Applied
Changed the logic to use `status != "running"` for expired condition and `status == "running"` for active condition. Both completed and failed runs are now eligible for purge once they pass the retention period.

**Files Modified:**
- `src/elspeth/core/retention/purge.py:84-103` - Updated `find_expired_row_payloads()` query
- `src/elspeth/core/retention/purge.py:137-152` - Updated `run_expired_condition` and `run_active_condition`

**Tests Added:**
- `tests/core/retention/test_purge.py::TestFailedRunsIncludedInPurge` - 3 new tests verifying:
  - Failed runs are eligible for purge after retention period
  - Failed runs don't protect shared refs from purge
  - Running runs still correctly protect their refs (regression test)
