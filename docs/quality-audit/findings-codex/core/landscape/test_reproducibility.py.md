# Test Defect Report

## Summary

- Assertion-free “no-op” test does not verify that `update_grade_after_purge` leaves the database unchanged for nonexistent run IDs.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/landscape/test_reproducibility.py:139` defines a test that only checks “no exception” and makes no assertions about side effects or database state.
- Code snippet from `tests/core/landscape/test_reproducibility.py:139`:
```python
def test_update_grade_after_purge_nonexistent_run_is_noop(self) -> None:
    """Non-existent run ID is handled gracefully (no error)."""
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.landscape.reproducibility import update_grade_after_purge

    db = LandscapeDB.in_memory()

    # Should not raise - just returns silently
    update_grade_after_purge(db, "nonexistent_run_id")
```
- Missing explicit checks (e.g., `get_run` returns `None` or runs table remains empty) are not present in `tests/core/landscape/test_reproducibility.py:146`.

## Impact

- The test will pass even if `update_grade_after_purge` erroneously creates a run record or mutates unrelated audit data.
- Regressions that introduce phantom runs or unintended updates could slip through without detection.
- Creates false confidence in “no-op” behavior for invalid run IDs.

## Root Cause Hypothesis

- Test author relied on “no exception” as sufficient for a no-op path and omitted explicit assertions about database state.

## Recommended Fix

- Add a direct assertion that no run was created and no records were modified (e.g., verify `recorder.get_run("nonexistent_run_id") is None` or assert runs table count stays zero).
- Example pattern:
```python
from elspeth.core.landscape.recorder import LandscapeRecorder

recorder = LandscapeRecorder(db)
assert recorder.get_run("nonexistent_run_id") is None
```
- Priority justification: P2 because this weak assertion can mask audit-integrity regressions around run creation or unintended mutation.
