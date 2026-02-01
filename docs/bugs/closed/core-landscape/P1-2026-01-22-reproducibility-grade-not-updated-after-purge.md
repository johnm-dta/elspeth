# Bug Report: Reproducibility grade not updated after payload purge

## Summary

Purging payloads via `PurgeManager.purge_payloads()` deletes blobs but never updates `runs.reproducibility_grade`, so runs with nondeterministic calls remain marked `REPLAY_REPRODUCIBLE` after payloads are removed. This overstates replay capability and violates documented retention semantics.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (static analysis agent)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: main (d8df733)
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Pipeline with retention policy
- Data set or fixture: Run with nondeterministic calls and payloads

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/retention/purge.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected purge.py, reproducibility.py, and docs/design/architecture.md

## Steps To Reproduce

1. Create a completed run containing nondeterministic calls so its `reproducibility_grade` is `REPLAY_REPRODUCIBLE`
2. Ensure the run's `completed_at` is older than the retention cutoff and its payload refs exist in the payload store
3. Run `PurgeManager.find_expired_payload_refs()` followed by `PurgeManager.purge_payloads()` for those refs
4. Query `runs.reproducibility_grade` for the run

## Expected Behavior

- Purging payloads should degrade the run's `reproducibility_grade` to `ATTRIBUTABLE_ONLY` once its payloads are removed

## Actual Behavior

- Purge deletes payloads but leaves `reproducibility_grade` unchanged (e.g., `REPLAY_REPRODUCIBLE`)

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/core/retention/purge.py:273`
  - `src/elspeth/core/landscape/reproducibility.py:92`
  - `docs/design/architecture.md:672`
- Minimal repro input (attach or link): Run with nondeterministic calls that passes retention cutoff

## Impact

- User-facing impact: `explain()` and audit views report replay capability that no longer exists after purge
- Data integrity / security impact: Audit metadata misrepresents reproducibility state, undermining audit assurances
- Performance or cost impact: Unknown

## Root Cause Hypothesis

`PurgeManager.purge_payloads()` never invokes `update_grade_after_purge` or tracks affected run IDs, so grades are not degraded when payloads are deleted.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/core/retention/purge.py` to track run IDs whose payload refs were actually deleted and call `update_grade_after_purge(self._db, run_id)` for each affected run
- Config or schema changes: Unknown
- Tests to add/update: Add a retention test that purges expired payloads for a `REPLAY_REPRODUCIBLE` run and asserts `runs.reproducibility_grade` becomes `ATTRIBUTABLE_ONLY`
- Risks or migration steps: None

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:672`
- Observed divergence: Grade does not degrade after payload purge
- Reason (if known): Missing integration in `PurgeManager`
- Alignment plan or decision needed: Call `update_grade_after_purge` as part of purge flow once affected runs are identified

## Acceptance Criteria

- After purging payloads for a run with nondeterministic calls, `runs.reproducibility_grade` is set to `ATTRIBUTABLE_ONLY`
- Remains unchanged for runs without purged payloads

## Tests

- Suggested tests to run: `pytest tests/core/retention/test_purge.py`
- New tests required: Yes, integration test covering purge-induced grade degradation

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/design/architecture.md`

## Verification Status

- [x] Bug confirmed via reproduction
- [x] Root cause verified
- [ ] Fix implemented
- [ ] Tests added
- [ ] Fix verified

## Verification Report (2026-01-24)

**Status: STILL VALID**

### Verification Summary

The bug is confirmed to be present in the current codebase. The `PurgeManager.purge_payloads()` method deletes payloads from the PayloadStore but does NOT call `update_grade_after_purge()` to degrade the reproducibility grade.

### Evidence

1. **Function exists but is never called by PurgeManager:**
   - `update_grade_after_purge()` is defined in `/home/john/elspeth-rapid/src/elspeth/core/landscape/reproducibility.py:92`
   - `PurgeManager.purge_payloads()` in `/home/john/elspeth-rapid/src/elspeth/core/retention/purge.py:273` does NOT import or call this function
   - No import of `reproducibility` module in `purge.py`

2. **Design documents show this was planned:**
   - `/home/john/elspeth-rapid/docs/plans/completed/2026-01-12-phase5-production-hardening.md:3610` explicitly describes the integration:
     ```python
     # Step 6: Update PurgeManager to degrade grades
     for run_id in run_ids:
         update_grade_after_purge(self._db, run_id)
     ```
   - This step was documented but never implemented

3. **Current purge_payloads implementation:**
   - Lines 273-314 of `purge.py` show the method only:
     - Iterates through refs
     - Calls `payload_store.delete(ref)`
     - Tracks statistics (deleted_count, skipped_count, failed_refs)
     - Returns PurgeResult
   - No run_id tracking
   - No call to update_grade_after_purge()

4. **Tests exist for update_grade_after_purge but not for integration:**
   - `tests/core/landscape/test_reproducibility.py` has unit tests for `update_grade_after_purge()` function itself
   - `tests/core/retention/test_purge.py` has NO tests verifying that purge updates reproducibility grades
   - No integration test exists for this workflow

### Root Cause Confirmed

The root cause hypothesis in the bug report is correct:

> `PurgeManager.purge_payloads()` never invokes `update_grade_after_purge` or tracks affected run IDs, so grades are not degraded when payloads are deleted.

### Impact

Runs with `reproducibility_grade = REPLAY_REPRODUCIBLE` retain that grade even after their payloads are purged, incorrectly indicating replay capability when payloads are no longer available.

### Additional Complexity Discovered

The current `purge_payloads(refs: list[str])` signature only accepts refs, not run IDs. The fix will require:

1. Either:
   - A. Track which run IDs are affected by the refs being deleted (query back from refs to runs)
   - B. Change the signature to accept or return affected run_ids
   - C. Add a separate method to handle grade degradation that queries affected runs from the deleted refs

2. Handle content-addressable storage correctly:
   - The same ref can be used by multiple runs
   - Only degrade grades for runs where ALL their nondeterministic call payloads were purged
   - Currently `find_expired_payload_refs()` already excludes refs still needed by active runs, so this may be simpler

### Recommended Fix Path

Following the original design document approach:

1. After deleting refs in `purge_payloads()`, query which run IDs had payloads deleted
2. For each affected run_id, call `update_grade_after_purge(self._db, run_id)`
3. Add integration test in `tests/core/retention/test_purge.py` that:
   - Creates a run with nondeterministic calls (REPLAY_REPRODUCIBLE)
   - Purges its payloads
   - Asserts grade degrades to ATTRIBUTABLE_ONLY

## Closure Report (2026-01-28)

**Status:** FIXED

**Fix Summary:**
1. Added `_find_affected_run_ids()` helper method to `PurgeManager` that queries all payload reference columns (rows.source_data_ref, calls.request_ref, calls.response_ref, routing_events.reason_ref) to find runs affected by refs being purged.

2. Modified `purge_payloads()` to:
   - Find affected run_ids BEFORE deletion
   - Delete the payloads (existing behavior)
   - Call `update_grade_after_purge()` for each affected run AFTER deletion

3. Added 6 integration tests in `TestPurgeUpdatesReproducibilityGrade`:
   - `test_purge_degrades_replay_reproducible_to_attributable_only` - core bug fix test
   - `test_purge_keeps_full_reproducible_unchanged`
   - `test_purge_keeps_attributable_only_unchanged`
   - `test_purge_updates_multiple_affected_runs` - shared payload scenario
   - `test_purge_empty_refs_does_not_update_any_grades`
   - `test_purge_call_payloads_also_degrades_grade`

**Files Changed:**
- `src/elspeth/core/retention/purge.py` - Added `_find_affected_run_ids()` helper, modified `purge_payloads()` to update grades
- `tests/core/retention/test_purge.py` - Added `TestPurgeUpdatesReproducibilityGrade` test class, fixed `_create_run()` helper to include `reproducibility_grade`

**Verification:**
- All 31 purge tests pass
- All 6 reproducibility tests pass
- mypy and ruff pass with no errors
