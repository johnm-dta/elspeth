# Test Quality Review: test_repositories.py

## Summary

This test suite covers the critical repository layer that converts database rows to domain objects. While it correctly validates the "crash on invalid enum" behavior mandated by the Data Manifesto, it has **critical gaps** in Tier 1 trust enforcement, massive duplication vulnerabilities from inline dataclass definitions, and incomplete coverage of corruption scenarios. For code that guards audit trail integrity, these gaps are unacceptable.

## Poorly Constructed Tests

### Test: All tests using inline @dataclass definitions (entire file)
**Issue**: Massive code duplication with 100+ lines of repeated dataclass definitions that are mutation-prone and brittle.
**Evidence**: Each test redefines `RunRow`, `NodeRow`, etc. with 8-13 fields. Example from lines 51-65:
```python
@dataclass
class RunRow:
    run_id: str
    started_at: datetime
    config_hash: str
    settings_json: str
    canonical_version: str
    status: str  # String in DB
    completed_at: datetime | None = None
    reproducibility_grade: str | None = None
    export_status: str | None = None
    export_error: str | None = None
    exported_at: datetime | None = None
    export_format: str | None = None
    export_sink: str | None = None
```
This exact structure is copy-pasted across 3 tests in `TestRunRepository` alone (lines 51-65, 86-99, 121-134), with total duplication exceeding 150 lines across the file.
**Fix**: Extract shared fixture mock classes to module level or conftest. Use factory functions for variations.
**Priority**: P1

### Test: test_load_primitive_fields tests (lines 358-387, 392-422, 427-448)
**Issue**: These tests provide zero value - they verify that `repo.load(db_row)` copies fields from input to output, which is trivial constructor behavior, not repository logic.
**Evidence**:
```python
def test_load_primitive_fields(self) -> None:
    """Repository loads Row with all primitive fields."""
    # ... setup ...
    assert row.row_id == "row-123"
    assert row.row_index == 42
    assert row.source_data_ref == "payload://xyz"
```
This is testing Python's `dataclass` construction, not the repository's responsibility (enum conversion).
**Fix**: Delete these tests entirely. Repositories that don't convert enums need no unit tests - the behavior is trivial. If we must test primitive field mapping, use a single property-based test that verifies all repositories preserve all fields.
**Priority**: P2

### Test: Missing test for NULL in non-nullable enum fields
**Issue**: Tests verify crash on invalid enum strings but don't test NULL values in fields that shouldn't be NULL.
**Evidence**: `test_load_crashes_on_invalid_status` tests `status="invalid_garbage"` but there's no test for `status=None` when `status` is required. Per Data Manifesto: "NULL where unexpected = crash".
**Fix**: Add tests like:
```python
def test_load_crashes_on_null_required_enum(self) -> None:
    """Repository crashes when required enum field is NULL."""
    db_row = RunRow(..., status=None)  # type: ignore
    repo = RunRepository(session=None)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        repo.load(db_row)
```
Add for every repository with required enum fields (Run.status, Node.node_type, Edge.default_mode, etc.).
**Priority**: P0

### Test: No tests for type corruption beyond strings
**Issue**: Tests only verify string enum validation. No tests for type corruption like integer IDs becoming floats, datetime fields becoming strings, etc.
**Evidence**: All "crash on invalid" tests use wrong-value strings (`status="invalid_garbage"`), never wrong-type values (`status=42` or `row_index="not-an-int"`).
**Fix**: Add type corruption tests:
```python
def test_load_crashes_on_wrong_type_for_datetime(self) -> None:
    """Repository crashes when datetime field has wrong type."""
    db_row = RunRow(..., started_at="2024-01-01T00:00:00")  # String not datetime
    # Should crash when constructing Run dataclass
```
**Priority**: P0

### Test: No tests for missing required fields
**Issue**: No tests verify crash behavior when required fields are absent from database rows.
**Evidence**: All mock rows include all fields. No test tries `db_row` missing `run_id` or `node_type`.
**Fix**: Add tests that use mock rows with missing attributes to verify `AttributeError` is raised:
```python
def test_load_crashes_on_missing_required_field(self) -> None:
    """Repository crashes when required field missing from DB row."""
    @dataclass
    class IncompleteRunRow:
        run_id: str
        # Missing started_at, config_hash, etc.

    db_row = IncompleteRunRow(run_id="run-123")
    repo = RunRepository(session=None)
    with pytest.raises(AttributeError):
        repo.load(db_row)
```
**Priority**: P1

## Misclassified Tests

No misclassified tests. These are correctly categorized as unit tests - they test repository enum conversion in isolation with no database dependency.

## Infrastructure Gaps

### Gap: No shared fixtures for mock database rows
**Issue**: Each test manually constructs mock dataclass rows, leading to 150+ lines of duplication.
**Evidence**: `RunRow` defined identically in lines 51-65, 86-99, 121-134, 155-168. Same pattern for all 8 repositories.
**Fix**: Create fixture factory in conftest:
```python
# tests/core/landscape/conftest.py
from dataclasses import dataclass, make_dataclass
from datetime import datetime, UTC
from typing import Any

@pytest.fixture
def mock_row_factory():
    """Factory for creating mock database rows."""
    def _make_row(row_class_name: str, **fields: Any):
        # Build field list with proper types
        field_defs = [(k, type(v), v) for k, v in fields.items()]
        return make_dataclass(row_class_name, field_defs)(**fields)
    return _make_row

# Or simpler: use dict-like objects
class MockRow:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@pytest.fixture
def mock_run_row():
    return lambda **overrides: MockRow(
        run_id="run-123",
        started_at=datetime.now(UTC),
        config_hash="abc",
        settings_json="{}",
        canonical_version="1.0.0",
        status="running",
        **overrides
    )
```
**Priority**: P1

### Gap: No parametrized tests for enum exhaustiveness
**Issue**: Each repository's enum crash tests manually test one invalid value per enum. No verification that ALL valid enum values are handled.
**Evidence**: `test_load_crashes_on_invalid_status` uses `status="invalid_garbage"` but doesn't verify all valid values like `"running"`, `"completed"`, `"failed"` actually work.
**Fix**: Use `@pytest.mark.parametrize` to test all valid enum values:
```python
@pytest.mark.parametrize("status_str", ["running", "completed", "failed", "cancelled"])
def test_load_accepts_all_valid_statuses(status_str: str, mock_run_row) -> None:
    """Repository accepts all valid RunStatus enum values."""
    db_row = mock_run_row(status=status_str)
    repo = RunRepository(session=None)
    run = repo.load(db_row)
    assert run.status == RunStatus(status_str)
```
**Priority**: P2

### Gap: No property-based tests for field preservation
**Issue**: Repositories must preserve all fields exactly (except enum conversion). No property-based verification.
**Evidence**: Primitive field tests manually check 3 fields each. No exhaustive verification.
**Fix**: Use Hypothesis to generate random valid database rows and verify all fields are preserved:
```python
from hypothesis import given, strategies as st

@given(
    run_id=st.text(min_size=1),
    config_hash=st.text(min_size=1),
    status=st.sampled_from(["running", "completed", "failed"]),
)
def test_run_repository_preserves_all_fields(run_id, config_hash, status):
    """Repository preserves all fields during load."""
    db_row = MockRow(
        run_id=run_id,
        config_hash=config_hash,
        status=status,
        # ... all other fields
    )
    repo = RunRepository(session=None)
    run = repo.load(db_row)

    assert run.run_id == db_row.run_id
    assert run.config_hash == db_row.config_hash
    # ... assert all fields match
```
**Priority**: P3

### Gap: No integration tests with actual SQLAlchemy Row objects
**Issue**: All tests use mock dataclasses. No verification that repositories work with real `sqlalchemy.engine.Row` objects.
**Evidence**: All tests pass `session=None` and use custom `@dataclass` mocks.
**Fix**: Add integration tests in `tests/integration/test_repository_sqlalchemy.py` that insert real rows via SQLAlchemy and load them via repositories. This catches issues like SQLAlchemy rows being immutable or having different attribute access patterns.
**Priority**: P2

### Gap: No test for session parameter usage
**Issue**: All repositories accept a `session` parameter but tests pass `session=None` and never verify the session is used (or not used) correctly.
**Evidence**: Every test does `repo = RunRepository(session=None)` and never checks if `self.session` is accessed.
**Fix**: Either (a) remove `session` from `__init__` if it's not used in `load()`, or (b) add tests that verify repositories don't misuse the session:
```python
def test_load_does_not_execute_queries(self) -> None:
    """load() method does not execute SQL - it only converts in-memory rows."""
    mock_session = Mock()
    repo = RunRepository(session=mock_session)
    repo.load(mock_run_row())
    mock_session.execute.assert_not_called()  # load() is pure conversion
```
**Priority**: P2

## Positive Observations

1. **Correct crash behavior**: Tests correctly verify that invalid enum strings cause `ValueError`, per Data Manifesto requirement for Tier 1 data.

2. **Clear test names**: Test names explicitly state what they verify (e.g., `test_load_crashes_on_invalid_status`).

3. **Docstrings reference policy**: Tests cite "per Data Manifesto" in docstrings, linking behavior to requirements.

4. **Type hints present**: Test methods have return type annotations (`-> None`).

5. **Focused scope**: Tests are properly scoped to repository responsibilities (enum conversion) rather than testing dataclass construction in depth.

## Critical Missing Test Coverage

The following corruption scenarios are **completely untested**:

| Corruption Scenario | Current Coverage | Required Test |
|---------------------|------------------|---------------|
| NULL in required enum field | ❌ None | Verify crash on `status=None` |
| Wrong type for enum (int/float) | ❌ None | Verify crash on `status=42` |
| Wrong type for datetime | ❌ None | Verify crash on `started_at="2024-01-01"` |
| Wrong type for int | ❌ None | Verify crash on `row_index="42"` |
| Missing required field | ❌ None | Verify crash on row without `run_id` |
| Extra unexpected fields | ❌ None | Verify behavior when DB row has unknown attrs |
| Case sensitivity of enum strings | ❌ None | Does `status="RUNNING"` work or crash? |
| Empty string enum values | ❌ None | Verify crash on `status=""` |

**Recommendation**: Add a dedicated test class for each repository with systematic corruption tests:

```python
class TestRunRepositoryCorruption:
    """Systematic corruption testing for RunRepository per Data Manifesto."""

    def test_null_in_required_enum_crashes(self): ...
    def test_wrong_type_for_enum_crashes(self): ...
    def test_wrong_type_for_datetime_crashes(self): ...
    def test_missing_required_field_crashes(self): ...
    def test_empty_string_enum_crashes(self): ...
```

## Summary of Priorities

**P0 (Critical - Fix Immediately):**
- Add NULL validation tests for required enum fields
- Add type corruption tests (wrong types for datetime, int, enum fields)

**P1 (High - Fix Soon):**
- Extract shared mock row fixtures to eliminate 150+ lines of duplication
- Add missing required field tests

**P2 (Medium - Should Fix):**
- Add parametrized tests for enum value exhaustiveness
- Add integration tests with real SQLAlchemy Row objects
- Clarify/test session parameter usage

**P3 (Low - Nice to Have):**
- Add property-based tests for field preservation
- Remove trivial primitive field tests (or replace with one property test)

**Overall Assessment**: This test suite covers the happy path and basic error cases but has **critical gaps** in Tier 1 trust enforcement. For code guarding audit trail integrity, the missing NULL/type corruption tests are unacceptable. The massive duplication from inline dataclasses is a maintenance nightmare waiting to happen.
