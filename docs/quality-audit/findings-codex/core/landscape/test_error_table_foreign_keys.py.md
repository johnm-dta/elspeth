# Test Defect Report

## Summary

- Missing negative coverage for `run_id` foreign key enforcement in error tables; tests only exercise token/node FKs.

## Severity

- Severity: major
- Priority: P1

## Category

- Incomplete Contract Coverage

## Evidence

- `src/elspeth/core/landscape/schema.py:311` and `src/elspeth/core/landscape/schema.py:330` define `run_id` as a non-null FK for both error tables.
- `tests/core/landscape/test_error_table_foreign_keys.py:58-66` and `tests/core/landscape/test_error_table_foreign_keys.py:319-327` always insert with `run_id=run.run_id`; there is no test that attempts an invalid `run_id` to verify FK enforcement.

```python
# src/elspeth/core/landscape/schema.py:311,330
Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False)

# tests/core/landscape/test_error_table_foreign_keys.py:58-66
transform_errors_table.insert().values(
    error_id="terr_orphan001",
    run_id=run.run_id,
    token_id="nonexistent_token",
    transform_id="node_test",
    row_hash="abc123",
    destination="discard",
    created_at=datetime.now(UTC),
)
```

## Impact

- A regression that removes or loosens the `run_id` FK would go undetected.
- Orphan error records could be created without a corresponding run, weakening audit lineage.
- Creates false confidence that all error-table FKs are enforced.

## Root Cause Hypothesis

- Tests were scoped to the original bug (token/node FKs) and did not expand to cover `run_id`.

## Recommended Fix

- Add negative tests in this file for invalid `run_id` in both tables, asserting `IntegrityError`.
- Example:

```python
def test_rejects_orphan_run_id_transform_errors(...):
    # create a valid token + transform node under a real run
    with pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"):
        with landscape_db.connection() as conn:
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_orphan_run",
                    run_id="missing_run",
                    token_id=token.token_id,
                    transform_id="node_test",
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()
```

- Priority justification: `run_id` is a core audit linkage; missing enforcement undermines Tier 1 integrity.
---
# Test Defect Report

## Summary

- Repeated run/node/token setup across tests; fixture duplication inflates maintenance and risk of inconsistent updates.

## Severity

- Severity: minor
- Priority: P2

## Category

- Fixture Duplication

## Evidence

- `tests/core/landscape/test_error_table_foreign_keys.py:73-96` and `tests/core/landscape/test_error_table_foreign_keys.py:118-142` repeat the same run/source/row/token setup, with similar duplication in `tests/core/landscape/test_error_table_foreign_keys.py:179-203` and `tests/core/landscape/test_error_table_foreign_keys.py:241-264`.

```python
# tests/core/landscape/test_error_table_foreign_keys.py:73-96
run = recorder.begin_run(config={"test": True}, canonical_version="1.0")
recorder.register_node(... node_id="source_test" ...)
row = recorder.create_row(...)
token = recorder.create_token(...)

# tests/core/landscape/test_error_table_foreign_keys.py:118-142
run = recorder.begin_run(config={"test": True}, canonical_version="1.0")
recorder.register_node(... node_id="source_test" ...)
row = recorder.create_row(...)
token = recorder.create_token(...)
```

## Impact

- Increases maintenance cost when setup changes (e.g., new required fields).
- Raises risk of inconsistent or stale setup across tests.
- Slows iteration by repeating verbose boilerplate.

## Root Cause Hypothesis

- Tests were added quickly around a bug fix without extracting shared fixtures.

## Recommended Fix

- Extract shared setup into fixtures in this file (or `tests/core/landscape/conftest.py`) and reuse across tests.
- Example:

```python
@pytest.fixture
def run_with_source(recorder: LandscapeRecorder):
    run = recorder.begin_run(config={"test": True}, canonical_version="1.0")
    recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type="source",
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
        node_id="source_test",
        sequence=0,
    )
    return run

@pytest.fixture
def token_with_transform(recorder: LandscapeRecorder, run_with_source):
    row = recorder.create_row(
        run_id=run_with_source.run_id,
        source_node_id="source_test",
        row_index=1,
        data={"id": "test-1"},
    )
    token = recorder.create_token(row_id=row.row_id)
    recorder.register_node(
        run_id=run_with_source.run_id,
        plugin_name="test_transform",
        node_type="transform",
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
        node_id="node_test",
        sequence=0,
    )
    return run_with_source, token
```

- Priority justification: Fixture duplication is a known maintenance risk and aligns with P2 quality issues.
