# Bug Report: RunRepository masks invalid export_status values

## Summary

`RunRepository.load` treats falsy `export_status` values as None, so invalid values like "" bypass `ExportStatus` coercion and do not crash. This masks Tier 1 data corruption and misreports export status, violating the audit DB "crash on anomaly" rule.

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
- Config profile / env vars: Any
- Data set or fixture: Corrupted audit DB with invalid export_status

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of repositories.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): sandbox_mode=read-only, approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed repository, contracts, schema, and recorder mappings

## Steps To Reproduce

1. Create a mock row with `export_status=""` (or another falsy non-None value) and valid required fields
2. Call `RunRepository.load(mock_row)`
3. Inspect `Run.export_status`

## Expected Behavior

- Invalid `export_status` values should raise `ValueError` (or otherwise crash) during `ExportStatus` coercion
- Only `None` should map to `None`

## Actual Behavior

- `export_status` is returned as `None` with no exception, masking the invalid value

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/core/landscape/repositories.py:55`
  - `src/elspeth/core/landscape/recorder.py:338`
  - `src/elspeth/contracts/audit.py:43`
- Minimal repro input (attach or link): Mock row with `export_status=""`

## Impact

- User-facing impact: Export status can appear "unset" when the DB contains an invalid value
- Data integrity / security impact: Violates Tier 1 crash-on-anomaly rule by silently accepting corrupted audit data
- Performance or cost impact: Unknown

## Root Cause Hypothesis

`export_status` is gated by a truthiness check instead of an explicit `None` check, so falsy invalid values bypass enum validation.

## Proposed Fix

- Code changes (modules/files): Update `RunRepository.load` in `src/elspeth/core/landscape/repositories.py:55` to use `row.export_status is not None` before coercion
- Config or schema changes: None
- Tests to add/update: Add a test for empty-string `export_status` in `tests/core/landscape/test_repositories.py`
- Risks or migration steps: None

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:40` - "invalid enum value = crash"
- Observed divergence: Invalid audit DB values can be silently coerced to None instead of crashing
- Reason (if known): Unknown
- Alignment plan or decision needed: Enforce explicit None checks and allow invalid values to raise

## Acceptance Criteria

- `RunRepository.load` raises `ValueError` for `export_status=""` (or other non-None invalid values)
- Preserves `None` as `None`
- Valid strings coerce to `ExportStatus` enums correctly

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_repositories.py::TestRunRepository`
- New tests required: Add a case asserting empty-string `export_status` raises `ValueError`

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:32`

## Verification Status

- [x] Bug confirmed via reproduction
- [x] Root cause verified
- [x] Fix implemented
- [x] Tests added
- [x] Fix verified

## Resolution

**Fixed by:** Claude Opus 4.5 (via Claude Code)
**Date:** 2026-01-28
**Branch:** fix/rc1-bug-burndown-session-6

### Fix Applied

**File:** `src/elspeth/core/landscape/repositories.py` (line 67-68)

Changed from truthiness check to explicit None check:
```python
# Before (buggy):
export_status=ExportStatus(row.export_status) if row.export_status else None,

# After (fixed):
# Use explicit is not None check - empty string should raise, not become None (Tier 1)
export_status=ExportStatus(row.export_status) if row.export_status is not None else None,
```

### Test Added

**File:** `tests/core/landscape/test_repositories.py`

Added `test_load_crashes_on_empty_string_export_status()` which verifies:
- Empty string `""` raises `ValueError: '' is not a valid ExportStatus`
- Matches the Tier 1 "crash on anomaly" requirement from Data Manifesto

### Verification

- All 420 landscape tests pass
- All 5 RunRepository tests pass
- mypy: no type errors
- ruff: all checks passed

---

## Verification Report

**Verified by:** Claude Sonnet 4.5 (via Claude Code)
**Date:** 2026-01-24
**Status:** **STILL VALID**

### Findings

#### Code Review

**File:** `/home/john/elspeth-rapid/src/elspeth/core/landscape/repositories.py`
**Line 55:**
```python
export_status=ExportStatus(row.export_status) if row.export_status else None,
```

**Analysis:**
- Uses truthiness check (`if row.export_status`) instead of explicit None check
- Empty string `""` is falsy and would bypass `ExportStatus()` coercion
- Would silently convert `""` to `None` instead of raising `ValueError`
- **Bug is PRESENT in current code**

#### Related Bug Fix

A similar bug was fixed in `recorder.py` on 2026-01-21 in commit `57c57f5`:
- **Fixed file:** `src/elspeth/core/landscape/recorder.py` (lines 335, 376)
- **Fix pattern:** Changed from `if row.export_status` to `if row.export_status is not None`
- **Closed bug:** `P2-2026-01-19-recorder-export-status-enum-mismatch.md`

**However, the same pattern was NOT applied to `repositories.py`**

#### Test Coverage

**File:** `/home/john/elspeth-rapid/tests/core/landscape/test_repositories.py`

**Existing tests:**
- `test_load_converts_export_status_to_enum()` - tests valid string conversion ✓
- `test_load_handles_null_export_status()` - tests None handling ✓
- `test_load_crashes_on_invalid_status()` - tests invalid RunStatus enum ✓

**Missing test:**
- No test for empty string `export_status=""` that should raise `ValueError`
- Bug report's proposed test case is NOT present

#### Git History

- Last modification to `repositories.py`: commit `c786410` (ELSPETH RC-1)
- No subsequent fixes applied to this file
- The recorder.py fix in `57c57f5` did not include repositories.py

### Conclusion

**Status: STILL VALID**

The bug exists in the current codebase:
1. Line 55 uses truthiness check instead of `is not None`
2. No test exists for the empty string edge case
3. The fix applied to `recorder.py` was not applied to `repositories.py`

### Recommended Next Steps

1. Apply the same fix pattern used in `recorder.py`:
   ```python
   # Change from:
   export_status=ExportStatus(row.export_status) if row.export_status else None,

   # To:
   export_status=ExportStatus(row.export_status) if row.export_status is not None else None,
   ```

2. Add regression test in `tests/core/landscape/test_repositories.py`:
   ```python
   def test_load_crashes_on_empty_string_export_status(self) -> None:
       """Repository crashes on empty string export_status per Data Manifesto."""
       # Test with export_status=""
   ```

3. Ensure consistency across both `recorder.py` and `repositories.py` for all enum fields
