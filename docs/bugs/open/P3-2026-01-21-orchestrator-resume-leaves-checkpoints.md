# Bug Report: Resume early-exit leaves checkpoints behind

## Summary

- When resume finds no unprocessed rows, it completes the run but does not delete checkpoints, leaving stale recovery state in the database.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: resume runs where all rows already processed

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Create a failed run with checkpoints but no remaining unprocessed rows.
2. Call Orchestrator.resume().
3. Inspect checkpoints table after resume completes.

## Expected Behavior

- Successful resume should delete checkpoints, matching normal completion behavior.

## Actual Behavior

- Resume returns early without deleting checkpoints.

## Evidence

- Early return skips _delete_checkpoints in `src/elspeth/engine/orchestrator.py:1132-1142`.
- Normal completion deletes checkpoints in run().

## Impact

- User-facing impact: stale checkpoints remain in DB after successful resume.
- Data integrity / security impact: recovery metadata is inconsistent with completed status.
- Performance or cost impact: unnecessary checkpoint storage.

## Root Cause Hypothesis

- Early-exit branch lacks checkpoint cleanup.

## Proposed Fix

- Code changes (modules/files):
  - Call _delete_checkpoints(run_id) before returning in the early-exit branch.
- Config or schema changes: N/A
- Tests to add/update:
  - Resume test where unprocessed_rows is empty and checkpoints are cleared.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): checkpoint cleanup on successful completion.
- Observed divergence: completed run retains checkpoints.
- Reason (if known): missing cleanup in early return.
- Alignment plan or decision needed: ensure resume mirrors run() completion cleanup.

## Acceptance Criteria

- Resume early-exit deletes checkpoints when run completes.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k resume -v`
- New tests required: yes, resume checkpoint cleanup test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md checkpointing
