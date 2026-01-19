# Error Event Persistence Verification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify that QuarantineEvent and TransformErrorEvent are fully persisted and queryable via explain()

**Architecture:** The implementation already exists (discovered during codebase exploration). This plan adds integration tests to verify the end-to-end flow: source validation failure → DB record → explain() query, and transform error → DB record → explain() query.

**Tech Stack:** pytest, SQLAlchemy, existing landscape infrastructure

**Status Note:** Original audit marked SDA-015, SDA-029-031 as PARTIAL. Explore agents found full implementation exists. This plan verifies correctness.

---

## Task 1: Verify Validation Error Recording Integration

**Files:**
- Test: `tests/integration/test_error_event_persistence.py` (create)
- Reference: `src/elspeth/plugins/context.py:119-172`
- Reference: `src/elspeth/core/landscape/recorder.py:1898-1941`
- Reference: `src/elspeth/core/landscape/schema.py:276-288`

**Step 1: Write the failing test for validation error persistence**

```python
"""Integration tests for error event persistence in landscape."""

import pytest
from sqlalchemy import select

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import validation_errors_table
from elspeth.plugins.context import PluginContext


class TestValidationErrorPersistence:
    """Verify validation errors are persisted to landscape database."""

    def test_validation_error_persisted_to_database(
        self, tmp_path, landscape_db: LandscapeDB, recorder: LandscapeRecorder
    ):
        """Validation error from source should be queryable in database."""
        # Arrange: Create a run
        run_id = recorder.create_run(
            config_hash="test-config-hash",
            settings_json='{"test": true}',
        ).run_id

        # Create context with landscape
        ctx = PluginContext(
            run_id=run_id,
            landscape=recorder,
            node_id="source_node",
        )

        # Act: Record a validation error
        error_token = ctx.record_validation_error(
            row={"id": "row-1", "bad_field": "not_an_int"},
            error="Field 'bad_field' expected int, got str",
            schema_mode="strict",
            destination="quarantine_sink",
        )

        # Assert: Error is in database
        with landscape_db.connection() as conn:
            result = conn.execute(
                select(validation_errors_table).where(
                    validation_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.run_id == run_id
        assert result.node_id == "source_node"
        assert "bad_field" in result.error
        assert result.schema_mode == "strict"
        assert result.destination == "quarantine_sink"
```

**Step 2: Run test to verify current state**

Run: `pytest tests/integration/test_error_event_persistence.py::TestValidationErrorPersistence::test_validation_error_persisted_to_database -v`

Expected: PASS (implementation exists)

If FAIL: The implementation has a bug that needs fixing.

**Step 3: Add test for discard destination**

```python
    def test_validation_error_with_discard_still_recorded(
        self, landscape_db: LandscapeDB, recorder: LandscapeRecorder
    ):
        """Even 'discard' destination records QuarantineEvent for audit."""
        run_id = recorder.create_run(
            config_hash="test-hash",
            settings_json="{}",
        ).run_id

        ctx = PluginContext(
            run_id=run_id,
            landscape=recorder,
            node_id="source_node",
        )

        # Act: Record with discard destination
        error_token = ctx.record_validation_error(
            row={"id": "discarded-row"},
            error="Missing required field",
            schema_mode="strict",
            destination="discard",
        )

        # Assert: Still recorded (audit completeness)
        with landscape_db.connection() as conn:
            result = conn.execute(
                select(validation_errors_table).where(
                    validation_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.destination == "discard"
```

**Step 4: Run tests**

Run: `pytest tests/integration/test_error_event_persistence.py::TestValidationErrorPersistence -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/test_error_event_persistence.py
git commit -m "$(cat <<'EOF'
test(integration): add validation error persistence tests

Verify that QuarantineEvent records are persisted to landscape
database and queryable. Confirms SDA-029 implementation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Verify Transform Error Recording Integration

**Files:**
- Modify: `tests/integration/test_error_event_persistence.py`
- Reference: `src/elspeth/plugins/context.py:174-223`
- Reference: `src/elspeth/core/landscape/recorder.py:1945-1987`
- Reference: `src/elspeth/core/landscape/schema.py:295-307`

**Step 1: Add transform error persistence test**

```python
from elspeth.core.landscape.schema import transform_errors_table


class TestTransformErrorPersistence:
    """Verify transform errors are persisted to landscape database."""

    def test_transform_error_persisted_to_database(
        self, landscape_db: LandscapeDB, recorder: LandscapeRecorder
    ):
        """Transform error should be queryable in database."""
        # Arrange
        run_id = recorder.create_run(
            config_hash="test-hash",
            settings_json="{}",
        ).run_id

        ctx = PluginContext(
            run_id=run_id,
            landscape=recorder,
            node_id="transform_node",
        )

        # Act: Record a transform error
        error_token = ctx.record_transform_error(
            token_id="token-123",
            transform_id="price_calculator",
            row={"quantity": 0, "total": 100},
            error_details={"reason": "division_by_zero", "field": "quantity"},
            destination="failed_calculations",
        )

        # Assert: Error is in database
        with landscape_db.connection() as conn:
            result = conn.execute(
                select(transform_errors_table).where(
                    transform_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.run_id == run_id
        assert result.token_id == "token-123"
        assert result.transform_id == "price_calculator"
        assert "division_by_zero" in result.error_details_json
        assert result.destination == "failed_calculations"

    def test_transform_error_with_discard_still_recorded(
        self, landscape_db: LandscapeDB, recorder: LandscapeRecorder
    ):
        """Even 'discard' destination records TransformErrorEvent."""
        run_id = recorder.create_run(
            config_hash="test-hash",
            settings_json="{}",
        ).run_id

        ctx = PluginContext(
            run_id=run_id,
            landscape=recorder,
            node_id="transform_node",
        )

        error_token = ctx.record_transform_error(
            token_id="token-456",
            transform_id="validator",
            row={"data": "invalid"},
            error_details={"reason": "validation_failed"},
            destination="discard",
        )

        with landscape_db.connection() as conn:
            result = conn.execute(
                select(transform_errors_table).where(
                    transform_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.destination == "discard"
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_error_event_persistence.py::TestTransformErrorPersistence -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_error_event_persistence.py
git commit -m "$(cat <<'EOF'
test(integration): add transform error persistence tests

Verify that TransformErrorEvent records are persisted to landscape
database and queryable. Confirms SDA-015 implementation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Verify explain() Returns Error Events

**Files:**
- Modify: `tests/integration/test_error_event_persistence.py`
- Reference: `src/elspeth/core/landscape/lineage.py:50-124`

**Step 1: Add explain() integration test**

```python
from elspeth.core.landscape.lineage import LineageQuery


class TestErrorEventExplainQuery:
    """Verify explain() includes error events in lineage."""

    def test_explain_includes_validation_errors(
        self, landscape_db: LandscapeDB, recorder: LandscapeRecorder
    ):
        """explain() should return validation error for quarantined row."""
        # Arrange: Create run and record validation error
        run_id = recorder.create_run(
            config_hash="test-hash",
            settings_json="{}",
        ).run_id

        ctx = PluginContext(
            run_id=run_id,
            landscape=recorder,
            node_id="csv_source",
        )

        error_token = ctx.record_validation_error(
            row={"id": "row-42", "value": "not_a_number"},
            error="Expected int for 'value'",
            schema_mode="strict",
            destination="quarantine",
        )

        # Act: Query lineage for this row
        lineage_query = LineageQuery(landscape_db)
        lineage = lineage_query.explain(
            run_id=run_id,
            row_id=error_token.row_id,
        )

        # Assert: Lineage includes validation error
        assert lineage is not None
        assert lineage.validation_error is not None
        assert lineage.validation_error.error_id == error_token.error_id
        assert "Expected int" in lineage.validation_error.error

    def test_explain_includes_transform_errors(
        self, landscape_db: LandscapeDB, recorder: LandscapeRecorder
    ):
        """explain() should return transform error for failed row."""
        run_id = recorder.create_run(
            config_hash="test-hash",
            settings_json="{}",
        ).run_id

        # First create a row and token
        row = recorder.record_row(
            run_id=run_id,
            row_index=0,
            row_id="row-99",
            source_data={"id": "row-99", "divisor": 0},
        )
        token = recorder.create_token(
            run_id=run_id,
            row_id=row.row_id,
        )

        ctx = PluginContext(
            run_id=run_id,
            landscape=recorder,
            node_id="divide_transform",
        )

        error_token = ctx.record_transform_error(
            token_id=token.token_id,
            transform_id="divide_transform",
            row={"id": "row-99", "divisor": 0},
            error_details={"reason": "division_by_zero"},
            destination="error_sink",
        )

        # Act: Query lineage
        lineage_query = LineageQuery(landscape_db)
        lineage = lineage_query.explain(
            run_id=run_id,
            token_id=token.token_id,
        )

        # Assert: Lineage includes transform error
        assert lineage is not None
        assert lineage.transform_error is not None
        assert lineage.transform_error.error_id == error_token.error_id
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_error_event_persistence.py::TestErrorEventExplainQuery -v`

Expected: PASS if explain() already queries error tables, FAIL if not.

**Step 3: If tests fail, add error event queries to LineageQuery**

If the tests fail because `LineageQuery.explain()` doesn't query error tables yet, this becomes implementation work:

Modify: `src/elspeth/core/landscape/lineage.py`

Add queries for `validation_errors_table` and `transform_errors_table` in the `explain()` method, joining on `run_id`/`row_id` or `token_id`.

**Step 4: Run full test suite**

Run: `pytest tests/integration/test_error_event_persistence.py -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add tests/integration/test_error_event_persistence.py src/elspeth/core/landscape/lineage.py
git commit -m "$(cat <<'EOF'
test(integration): add explain() error event verification

Verify that explain() queries include validation and transform
errors for complete audit lineage. Closes go-live blocker SDA-029-031.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add Conftest Fixtures

**Files:**
- Create or Modify: `tests/integration/conftest.py`

**Step 1: Add shared fixtures if not present**

```python
"""Shared fixtures for integration tests."""

import pytest
from pathlib import Path

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


@pytest.fixture
def landscape_db(tmp_path: Path) -> LandscapeDB:
    """Create a temporary landscape database for testing."""
    db_path = tmp_path / "test_landscape.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    db.create_tables()
    return db


@pytest.fixture
def recorder(landscape_db: LandscapeDB) -> LandscapeRecorder:
    """Create a LandscapeRecorder with the test database."""
    return LandscapeRecorder(landscape_db)
```

**Step 2: Run all tests**

Run: `pytest tests/integration/test_error_event_persistence.py -v`

Expected: All PASS

**Step 3: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_error_event_persistence.py
git commit -m "$(cat <<'EOF'
test(integration): add conftest fixtures for error event tests

Add shared landscape_db and recorder fixtures for integration tests.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan verifies that the error event persistence implementation (which already exists) works correctly end-to-end:

1. ✅ Validation errors recorded to `validation_errors_table`
2. ✅ Transform errors recorded to `transform_errors_table`
3. ✅ Both work with "discard" destination (audit completeness)
4. ✅ `explain()` can query error events for complete lineage

**Expected Outcome:** All tests pass, confirming SDA-015, SDA-029, SDA-030, SDA-031 are fully implemented.

**If Tests Fail:** The failure points to specific bugs in the existing implementation that need fixing.
