# Bug Report: Plugin cleanup exceptions suppressed, hiding system bugs

## Summary

- Orchestrator suppresses exceptions from `on_complete()` and `close()` during run and resume cleanup, violating the "plugin bugs must crash" rule and allowing defective plugins to pass silently.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-30
- Related run/issue ID: N/A

## Environment

- Commit/branch: 2678b83d1bef5b1ab2049b9babe625f4fb0b2799 (fix/P2-aggregation-metadata-hardcoded)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Steps To Reproduce

1. Create a transform or sink whose `on_complete()` raises an exception.
2. Run a pipeline with that plugin.
3. Observe the run completes or fails for other reasons without surfacing the `on_complete()` exception.
4. (Resume path) Create a transform whose `close()` raises and resume a run; observe the exception is suppressed.

## Expected Behavior

- Plugin lifecycle exceptions should crash the run/resume (or be aggregated and raised after all cleanup attempts), per system-owned plugin contract.

## Actual Behavior

- Exceptions in `on_complete()` and resume `close()` are swallowed, allowing runs to proceed without surfacing plugin bugs.

## Evidence

- Suppressed `on_complete()` exceptions in main run cleanup: `src/elspeth/engine/orchestrator.py:1649-1657`.
- Suppressed `on_complete()` and `close()` exceptions in resume cleanup: `src/elspeth/engine/orchestrator.py:2520-2535`.

## Impact

- User-facing impact: Silent plugin failures; run appears successful or fails for unrelated reasons without indicating cleanup failures.
- Data integrity / security impact: Plugin bugs can persist undetected; may leave partial outputs or inconsistent state with no clear audit signal.
- Performance or cost impact: Potential resource leaks (connections/files) without error reporting.

## Root Cause Hypothesis

- Cleanup paths use `suppress(Exception)` to continue teardown, but never re-raise aggregated errors, violating the "plugin exceptions must crash" rule.

## Proposed Fix

- Code changes (modules/files):
  - In `src/elspeth/engine/orchestrator.py`, replace `suppress(Exception)` blocks for `on_complete()` and resume `close()` with collection + raise (similar to `_cleanup_transforms()`), so failures are surfaced after attempting all cleanups.
- Config or schema changes: None.
- Tests to add/update:
  - Add a unit/integration test where a transform's `on_complete()` raises and assert the run fails.
  - Add a resume-path test where a transform's `close()` raises and assert resume fails.
- Risks or migration steps:
  - Runs may now fail on cleanup bugs that were previously hidden; this is desired and aligns with auditability requirements.

## Acceptance Criteria

- If any plugin `on_complete()` or resume `close()` raises, the run/resume fails with a clear exception after all cleanup attempts are made.
- Tests demonstrate that cleanup exceptions are no longer suppressed.

## Verification (2026-02-01)

**Status: STILL VALID**

- Main run cleanup still wraps `on_complete()` calls in `suppress(Exception)` with no re-raise. (`src/elspeth/engine/orchestrator.py:1649-1657`)
- Resume cleanup still suppresses `on_complete()` and `close()` exceptions. (`src/elspeth/engine/orchestrator.py:2520-2535`)

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/unit/ -k "on_complete_exception or resume_close_exception"`
- New tests required: yes, as described above.

## Resolution (2026-02-02)

**Status: ALREADY FIXED**

Investigation revealed this bug was fixed prior to the 2026-02-01 verification (stale verification).

### Current Implementation

The orchestrator now correctly:
1. Collects cleanup errors into a list
2. Attempts ALL cleanups before failing (best-effort pattern)
3. Re-raises aggregated errors after all cleanup attempts

**Code locations:**
- Main run: `orchestrator.py:1720-1724` - raises `RuntimeError(f"Plugin cleanup failed: {error_summary}")`
- Resume: `orchestrator.py:2632-2636` - same pattern
- `_cleanup_transforms()`: `orchestrator.py:289-291` - same pattern

### Existing Test Coverage

Tests already verify this behavior:

| Test | File | Assertion |
|------|------|-----------|
| `test_cleanup_continues_if_one_close_fails` | `test_orchestrator_cleanup.py` | `pytest.raises(RuntimeError, match="Plugin cleanup failed")` |
| `test_transform_close_called_when_on_complete_fails` | `test_orchestrator_resume.py` | `pytest.raises(RuntimeError, match="Plugin cleanup failed")` |

### Verification

```bash
.venv/bin/python -m pytest tests/engine/test_orchestrator_cleanup.py tests/engine/test_orchestrator_resume.py::TestOrchestratorResumeCleanup -v
# 6 passed
```

No code changes required - bug was already fixed.
