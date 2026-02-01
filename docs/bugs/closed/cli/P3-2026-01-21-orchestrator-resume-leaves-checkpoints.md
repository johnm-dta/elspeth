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

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 4

**Current Code Analysis:**

The bug is **still present** in the current codebase. Examining `/home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py`:

**Early-exit path (lines 1337-1347):**
```python
if not unprocessed_rows:
    # All rows were processed - complete the run
    recorder.complete_run(run_id, status="completed")
    return RunResult(
        run_id=run_id,
        status=RunStatus.COMPLETED,
        rows_processed=0,
        rows_succeeded=0,
        rows_failed=0,
        rows_routed=0,
    )
```
**Missing:** No call to `self._delete_checkpoints(run_id)` before the early return.

**Normal completion path (lines 1361-1368):**
```python
# 6. Complete the run
recorder.complete_run(run_id, status="completed")
result.status = RunStatus.COMPLETED

# 7. Delete checkpoints on successful completion
self._delete_checkpoints(run_id)

return result
```
**Present:** Checkpoint deletion happens correctly in the normal path.

**Git History:**

Both code paths were introduced in commit `c786410` (ELSPETH - Release Candidate 1, 2026-01-22). The early-exit path has never included checkpoint deletion. No subsequent commits have modified the early-exit logic to add checkpoint cleanup.

Relevant commits since RC1:
- `b2a3518` (2026-01-23): "fix(sources,resume): comprehensive data handling bug fixes" - addressed type fidelity in resume but did not touch checkpoint cleanup
- No other commits have modified the early-exit path in `resume()`

**Root Cause Confirmed:**

Yes, the bug is confirmed. The `resume()` method has two completion paths:

1. **Early-exit path:** When `unprocessed_rows` is empty (all rows already processed), the method completes the run and returns immediately without calling `_delete_checkpoints(run_id)`.

2. **Normal path:** When rows are processed, the method completes the run and calls `_delete_checkpoints(run_id)` before returning.

This inconsistency means that if a resume operation finds no work to do (e.g., all rows were already successfully processed before the checkpoint was taken, or recovery was run multiple times), the checkpoints remain in the database even though the run is marked as completed.

**Impact:**
- Stale checkpoint metadata persists in the database
- Recovery metadata is inconsistent with run status (completed run still has checkpoints)
- Minor storage waste (checkpoints table grows unnecessarily)
- No functional impact on correctness (stale checkpoints won't break future operations)

**Test Coverage:**

No existing test covers the early-exit scenario. The test file `/home/john/elspeth-rapid/tests/engine/test_orchestrator_resume.py` only tests cases where unprocessed rows exist and are successfully processed. There is no test for the case where `get_unprocessed_row_data()` returns an empty list.

**Recommendation:**

**Keep open** - This is a valid bug that should be fixed. The fix is straightforward:

```python
if not unprocessed_rows:
    # All rows were processed - complete the run
    recorder.complete_run(run_id, status="completed")
    # Delete checkpoints on successful completion (matches normal path)
    self._delete_checkpoints(run_id)
    return RunResult(
        run_id=run_id,
        status=RunStatus.COMPLETED,
        rows_processed=0,
        rows_succeeded=0,
        rows_failed=0,
        rows_routed=0,
    )
```

A test should also be added to verify checkpoint cleanup in the early-exit case.
