# Test Defect Report

## Summary

- Tests assert only on keys of the local dict literals, so they do not validate the TypedDict contracts or the `total=False` behavior they claim to cover.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/test_update_schemas.py:13` The test asserts against the dict it just created, not the TypedDict schema, so it will pass even if the contract changes or required keys are introduced.
```python
# Should accept all valid fields
update: ExportStatusUpdate = {
    "export_status": ExportStatus.COMPLETED,
    "exported_at": datetime.now(UTC),
}
assert "export_status" in update
```
- `tests/contracts/test_update_schemas.py:30` The "partial updates" test never inspects `ExportStatusUpdate` required/optional keys, so it cannot detect if `total=False` is removed.
```python
update: ExportStatusUpdate = {"export_status": ExportStatus.PENDING}
assert len(update) == 1
```
- `src/elspeth/contracts/audit.py:366` The contracts define multiple optional fields and rely on `total=False`, but none of that is asserted in the tests.
```python
class ExportStatusUpdate(TypedDict, total=False):
    export_status: ExportStatus
    exported_at: datetime
    export_error: str
    export_format: str
    export_sink: str
```

## Impact

- Contract regressions (renamed fields, removed fields, or a switch to `total=True`) will not fail these tests.
- Tests provide false confidence that update schemas are enforced or even checked at runtime.
- Downstream code could silently drift from the intended contract without test signal.

## Root Cause Hypothesis

- Misunderstanding that `TypedDict` has no runtime enforcement, leading to assertions that only validate local dict literals.

## Recommended Fix

- Assert the actual TypedDict schema via runtime introspection, not the local dict values:
  - Check `__required_keys__` is empty and `__optional_keys__` matches the expected field set.
  - Use `typing.get_type_hints` to assert field types (e.g., `ExportStatus`, `datetime`, `str`).
- Example pattern to add to `tests/contracts/test_update_schemas.py`:
```python
from typing import get_type_hints

def test_export_status_update_schema(self) -> None:
    from elspeth.contracts import ExportStatusUpdate, ExportStatus

    assert ExportStatusUpdate.__required_keys__ == set()
    assert ExportStatusUpdate.__optional_keys__ == {
        "export_status",
        "exported_at",
        "export_error",
        "export_format",
        "export_sink",
    }
    hints = get_type_hints(ExportStatusUpdate)
    assert hints["export_status"] is ExportStatus
```
- Priority justification: These are contract tests; schema drift should be caught early, and the fix is local to the test file.
