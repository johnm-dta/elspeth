# Bug Report: Resume path never calls transform.close()

## Summary

- _process_resumed_rows() calls on_complete for transforms but never calls close(), so transform resources (threads, clients, executors) are leaked during resume runs.

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
- Data set or fixture: resume runs with transforms that allocate resources

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a transform that allocates resources (e.g., pooled executor).
2. Run a pipeline, force a failure, and resume.
3. Observe that transform.close() is not called during resume cleanup.

## Expected Behavior

- Resume path should call transform.close() just like normal runs.

## Actual Behavior

- Only on_complete is called; close() is skipped in resume cleanup.

## Evidence

- Resume cleanup closes sinks only; no transform.close in `src/elspeth/engine/orchestrator.py:1424-1435`.
- Normal run cleanup calls transform.close via _cleanup_transforms().

## Impact

- User-facing impact: possible resource leaks or lingering threads after resume.
- Data integrity / security impact: none direct.
- Performance or cost impact: increased memory/CPU usage on repeated resumes.

## Root Cause Hypothesis

- Resume cleanup path omitted transform.close() call.

## Proposed Fix

- Code changes (modules/files):
  - Add transform.close() calls in _process_resumed_rows() finally block (mirror _cleanup_transforms()).
- Config or schema changes: N/A
- Tests to add/update:
  - Resume test that asserts transform.close() is invoked.
- Risks or migration steps:
  - Ensure close() is idempotent (contract already requires this).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): TransformProtocol requires close() idempotency and lifecycle cleanup.
- Observed divergence: resume path omits close().
- Reason (if known): missing cleanup in resume path.
- Alignment plan or decision needed: standardize cleanup across run and resume.

## Acceptance Criteria

- Resumed runs invoke transform.close() for all transforms.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k resume -v`
- New tests required: yes, resume cleanup test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md lifecycle hooks

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 5

**Current Code Analysis:**

The bug is **confirmed valid** in the current codebase. Analysis of `/home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py`:

1. **Normal run path (run() method, line 413-577):**
   - Has a `finally` block at line 574-576 that calls `self._cleanup_transforms(config)`
   - The `_cleanup_transforms()` method (lines 196-216) properly calls `close()` on all transforms
   - This ensures resource cleanup even on failure

2. **Resume path (resume() method, lines 1278-1368):**
   - **NO finally block** - missing cleanup
   - **NO call to `_cleanup_transforms()`** anywhere in the method
   - Calls `_process_resumed_rows()` which has its own finally block (lines 1645-1656)
   - `_process_resumed_rows()` finally block only calls:
     - `transform.on_complete(ctx)` for all transforms (lines 1647-1649)
     - `sink.close()` for all sinks (lines 1654-1656)
     - **Does NOT call `transform.close()`**

3. **The asymmetry:**
   ```
   run() method:
     finally:
       _cleanup_transforms(config)  # Calls transform.close()

   resume() method:
     # NO finally block, NO _cleanup_transforms() call

   _process_resumed_rows():
     finally:
       # Calls on_complete() but NOT close()
       for transform in config.transforms:
         transform.on_complete(ctx)
       # Only sinks get close(), not transforms
       for sink in config.sinks.values():
         sink.close()
   ```

**Git History:**

Commit `9cc5063` (2026-01-15) added `_cleanup_transforms()` and wired it into the normal run path:
- Added the `_cleanup_transforms()` method
- Added finally block to `run()` calling `_cleanup_transforms(config)`
- Added comprehensive tests in `tests/engine/test_orchestrator_cleanup.py`
- **But did NOT update the resume path**

This confirms the bug was introduced when cleanup was added to run() but not to resume().

**Root Cause Confirmed:**

The resume path lacks the same cleanup infrastructure as the normal run path. Transforms that allocate resources (thread pools, HTTP clients, database connections, etc.) in `on_start()` and clean them up in `close()` will leak those resources during resumed runs.

The `_process_resumed_rows()` method calls `on_complete()` (which flushes buffers, completes work) but skips `close()` (which releases resources). This distinction is important:
- `on_complete()` = finalize work
- `close()` = release resources (idempotent, must always be called)

**Recommendation:**

**Keep open** - This is a valid resource leak bug that needs fixing.

**Suggested fix:**
1. Add a finally block to `resume()` method that calls `self._cleanup_transforms(config)`
2. Add test case in `tests/engine/test_orchestrator_resume.py` verifying transforms get `close()` called during resume (similar to `test_orchestrator_cleanup.py`)

This mirrors the pattern in `run()` and ensures parity between normal and resume paths.

---

## Resolution

**Fixed in:** 2026-01-28
**Fixed by:** Claude Code (Opus 4.5)

**Fix:** Added `transform.close()` calls to the finally block of `_process_resumed_rows()`:

**Code changes:**
- `src/elspeth/engine/orchestrator.py` (lines 2255-2262):
  - Added loop calling `transform.close()` with `suppress(Exception)` wrapper
  - Also wrapped `sink.close()` with `suppress(Exception)` for consistency
  - Mirrors the best-effort cleanup pattern from `_cleanup_transforms()`

**Before:**
```python
finally:
    for transform in config.transforms:
        with suppress(Exception):
            transform.on_complete(ctx)
    for sink in config.sinks.values():
        with suppress(Exception):
            sink.on_complete(ctx)
    for sink in config.sinks.values():
        sink.close()
```

**After:**
```python
finally:
    for transform in config.transforms:
        with suppress(Exception):
            transform.on_complete(ctx)
    for sink in config.sinks.values():
        with suppress(Exception):
            sink.on_complete(ctx)
    # Close all transforms (release resources)
    for transform in config.transforms:
        with suppress(Exception):
            transform.close()
    for sink in config.sinks.values():
        with suppress(Exception):
            sink.close()
```

**Tests added:**
- `tests/engine/test_orchestrator_resume.py`: Added `TestOrchestratorResumeCleanup` class with `test_transform_close_called_during_resume`

**Commits:**
- fix(orchestrator): call transform.close() during resume cleanup (P3-2026-01-28)
