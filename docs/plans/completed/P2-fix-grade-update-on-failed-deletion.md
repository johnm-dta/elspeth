# Implementation Plan: Fix Grade Update on Failed Deletion

**Bug:** P2-2026-01-28-grade-update-on-failed-deletion
**Estimated Time:** 15 minutes
**Complexity:** Low
**Risk:** Low

## Summary

When `payload_store.delete(ref)` returns `False` (e.g., permissions error, I/O failure), the `purge_payloads()` method still updated the reproducibility grade for ALL affected runs, incorrectly marking runs as `ATTRIBUTABLE_ONLY` despite their payloads still existing.

This is an audit integrity issue: the system would claim a run can't be replayed when it actually can.

## Root Cause

In `purge.py`, the code computed `affected_run_ids` from ALL refs BEFORE deletion, then updated grades for ALL runs regardless of deletion success:

```python
# OLD CODE (buggy):
affected_run_ids = self._find_affected_run_ids(refs)  # From ALL refs

for ref in refs:
    if self._payload_store.exists(ref):
        deleted = self._payload_store.delete(ref)
        if deleted:
            deleted_count += 1
        else:
            failed_refs.append(ref)  # Tracked but not excluded from grade update

# Grades updated for ALL affected runs, including those with only failed refs
for run_id in affected_run_ids:
    update_grade_after_purge(self._db, run_id)
```

Runs that only had failed deletions would have their grades incorrectly downgraded.

## Implementation

### Step 1: Track successfully deleted refs

**File:** `src/elspeth/core/retention/purge.py`

Added `deleted_refs: list[str] = []` to track refs that were actually deleted, and moved the `_find_affected_run_ids()` call to AFTER deletion, passing only `deleted_refs`:

```python
# NEW CODE (fixed):
deleted_refs: list[str] = []

for ref in refs:
    if self._payload_store.exists(ref):
        deleted = self._payload_store.delete(ref)
        if deleted:
            deleted_count += 1
            deleted_refs.append(ref)  # Track successful deletions
        else:
            failed_refs.append(ref)

# Only compute affected runs from SUCCESSFULLY deleted refs
affected_run_ids = self._find_affected_run_ids(deleted_refs)

for run_id in affected_run_ids:
    update_grade_after_purge(self._db, run_id)
```

### Step 2: Updated docstring

Added documentation clarifying the new behavior:

```python
"""
Grade updates only occur for runs whose payloads were actually deleted.
Runs that only had failed deletions retain their grade (payloads still exist).
"""
```

## Tests Added

**File:** `tests/core/retention/test_purge.py`

1. **`test_purge_does_not_degrade_grade_when_deletion_fails`** - Verifies that when ALL deletions fail, the run's grade remains `REPLAY_REPRODUCIBLE` (not downgraded to `ATTRIBUTABLE_ONLY`).

2. **`test_purge_degrades_grade_when_some_deletions_succeed`** - Verifies that when SOME deletions succeed and others fail for the same run, the grade IS downgraded (conservative approach - any successful deletion means partial replay loss).

## Testing Checklist

- [x] When all deletions fail, grade remains unchanged
- [x] When some deletions succeed, grade is downgraded
- [x] When all deletions succeed, grade is downgraded (existing behavior preserved)
- [x] All 33 purge tests pass
- [x] All 9 grade-related tests in test_recorder_grades.py pass
- [x] Lint passes

## Acceptance Criteria

1. ✅ Runs with ONLY failed ref deletions retain their `REPLAY_REPRODUCIBLE` grade
2. ✅ Runs with at least one successful deletion get downgraded to `ATTRIBUTABLE_ONLY`
3. ✅ No regression in existing purge behavior
4. ✅ Tests document the expected behavior for partial failures

## Notes

**Why this matters for audit integrity:**

Per CLAUDE.md, ELSPETH must never misrepresent what it knows. Claiming a run is `ATTRIBUTABLE_ONLY` (not replayable) when payloads actually exist is evidence misrepresentation. An auditor asking "can I replay this run?" would get a wrong answer.

**Why partial success still downgrades:**

If a run has multiple payloads (source data, LLM responses, routing reasons) and ANY are deleted, replay is incomplete. Conservative downgrade is correct - we don't claim replayability we can't deliver.

---

## Implementation Summary

**Status:** Completed
**Commits:** See git history for this feature
**Notes:** Fixed purge_payloads() to only update reproducibility grades for runs whose payloads were actually deleted, not for runs where deletion failed.
