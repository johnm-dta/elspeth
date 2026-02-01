# Bug Report: RecoveryManager.can_resume blocks resume for interrupted runs left in RUNNING status

## Summary

- `can_resume()` returns `can_resume=False` for runs with `RunStatus.RUNNING`, but crashed/interrupted runs stay in RUNNING status. This blocks the documented crash recovery workflow.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/checkpoint/recovery.py:90-91` - explicitly returns `can_resume=False` with reason "Run is still in progress" for RUNNING status
- `docs/runbooks/resume-failed-run.md:40` - documents that operators should resume runs with status `running` and no `completed_at` (crash scenario)

## Impact

- User-facing impact: Cannot resume crashed runs without manual database intervention
- Data integrity / security impact: Forces workarounds that bypass normal resume flow
- Performance or cost impact: Significant operator time to manually fix run status

## Root Cause Hypothesis

- `can_resume()` treats RUNNING as "actively running" but doesn't account for crashed processes that left runs in RUNNING state.

## Proposed Fix

- Code changes:
  - Check if run is actually active (e.g., heartbeat timestamp, process lock)
  - Or: Allow resume for RUNNING runs that have no `completed_at` and last activity > threshold
  - Or: Add `--force-resume` flag to override the RUNNING check
- Tests to add/update:
  - Add test for crashed run scenario (RUNNING status, stale activity), assert can_resume=True

## Acceptance Criteria

- Crashed runs (RUNNING status, no process holding lock) can be resumed
- Actually running pipelines are still protected from concurrent resume
