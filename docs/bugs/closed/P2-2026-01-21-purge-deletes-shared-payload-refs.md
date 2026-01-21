# Bug Report: Purge deletes payload refs still used by active runs

## Summary

`find_expired_payload_refs()` returned refs for deletion without checking if those refs were also used by non-expired runs. Because payloads are content-addressable (same hash = same blob), a purge could delete blobs needed by active runs, breaking replay/explain.

## Severity

- Severity: major (data loss)
- Priority: P2

## Reporter

- Name or handle: Review comment analysis
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1

## Steps To Reproduce

1. Run A (60 days ago) processes content X, stores payload hash H
2. Run B (10 days ago) processes SAME content X, reuses payload hash H (content-addressable)
3. Purge with retention_days=30 runs
4. `find_expired_payload_refs()` finds H (from Run A)
5. `purge_payloads([H])` deletes the blob
6. User tries to replay/explain Run B (only 10 days old)
7. `PayloadStore.retrieve(H)` fails with KeyError

## Expected Behavior

Shared payload refs should NOT be deleted if any non-expired run still needs them.

## Actual Behavior

Shared refs were returned for deletion because the query only checked if the ref appeared in expired runs, not if it ALSO appeared in active runs.

## Root Cause

`find_expired_payload_refs()` at `purge.py:145-151` built queries for refs from expired runs but never subtracted refs that were ALSO used by non-expired runs.

## Resolution

**Status:** FIXED (2026-01-21)

**Solution Applied:** Added anti-join pattern to exclude refs still used by active runs.

**Changes Made:**
1. `src/elspeth/core/retention/purge.py`: Rewrote `find_expired_payload_refs()` to:
   - Build queries for refs from EXPIRED runs
   - Build queries for refs from ACTIVE runs (recent, running, or failed)
   - Return set difference: `expired_refs - active_refs`

2. `tests/core/retention/test_purge.py`: Added `TestContentAddressableSharedRefs` class with 4 regression tests:
   - `test_shared_row_ref_excluded_when_used_by_recent_run`
   - `test_shared_call_ref_excluded_when_used_by_recent_run`
   - `test_exclusive_expired_ref_is_returned`
   - `test_shared_ref_excluded_when_used_by_running_run`

**Key Design Decision:** Used Python set difference rather than SQL EXCEPT because:
1. SQLite's EXCEPT can have performance issues with complex UNIONs
2. Result sets are typically small enough for in-memory operation
3. Python set operations are clearer for this anti-join pattern

**Verification:**
- All 21 purge tests pass
- Manual verification confirms shared refs are now correctly excluded
