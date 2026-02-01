# Test Quality Review: test_error_table_foreign_keys.py

## Summary

This test file provides comprehensive foreign key constraint verification for error tables with strong adherence to auditability principles. Tests correctly verify both rejection of orphan records and RESTRICT behavior for deletion. However, the test suite suffers from significant infrastructure waste (massive setup duplication), missing coverage of cascade scenarios, and lacks verification of actual FK constraint presence in the schema.

## Poorly Constructed Tests

### Test: test_accepts_valid_foreign_keys (line 238)
**Issue**: Incomplete assertion - only verifies record creation, not FK constraint existence
**Evidence**:
```python
assert error_record is not None
assert error_record.token_id == token.token_id
assert error_record.transform_id == "node_test"
```
The test proves insertion succeeded with valid values but doesn't verify that FK constraints are actually enforced. A database without FK constraints would also pass this test.
**Fix**: This is not actually testing FK behavior - it's testing basic INSERT. Either remove this test as redundant (the constraint tests already prove FKs exist), or rename it to `test_insert_transform_error_with_valid_references` and move it to a different test class focused on basic operations.
**Priority**: P2

### Test: test_accepts_valid_node_id (line 407)
**Issue**: Same as above - incomplete FK verification
**Evidence**: Only tests successful insertion, not constraint enforcement
**Fix**: Remove or rename to `test_insert_validation_error_with_valid_reference`
**Priority**: P2

### Test: All tests in both classes (lines 34-448)
**Issue**: Massive setup duplication violates DRY principle
**Evidence**: Every test contains this identical 12-18 line setup pattern:
```python
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
row = recorder.create_row(...)
token = recorder.create_token(...)
# ... repeat in every test
```
This setup appears in 9 different tests with only minor variations (node_id names, whether source is needed).
**Fix**: Extract fixtures for common scenarios:
- `@pytest.fixture` for `run_with_source_node`
- `@pytest.fixture` for `run_with_source_and_transform_nodes`
- `@pytest.fixture` for `token_with_transform_node` (returns tuple of token, transform node)
**Priority**: P1

## Misclassified Tests

### Tests: Entire test suite
**Issue**: Tests are correctly classified as unit tests for schema constraints, but missing integration tests for cascade behavior
**Evidence**: Tests directly manipulate tables via SQLAlchemy Core, testing FK constraints in isolation. This is correct for unit tests, but integration tests are needed.
**Fix**: This file is correctly unit-level. However, create a separate integration test file `test_error_table_cascades_integration.py` to verify:
1. When a run is deleted, do error records CASCADE properly (if that's the intent)?
2. When using recorder methods that delete nodes/tokens, do error records prevent deletion as expected?
3. Transaction rollback scenarios where FK violations should abort the entire transaction
**Priority**: P2

## Infrastructure Gaps

### Gap: No fixture for common entity setup
**Issue**: Every test manually creates runs, nodes, rows, tokens with identical parameters
**Evidence**: 9 tests contain copy-paste setup code (see "Poorly Constructed Tests" above)
**Fix**: Add to `tests/core/landscape/conftest.py`:
```python
@pytest.fixture
def run_with_source(recorder: LandscapeRecorder):
    """Create a run with a registered source node."""
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
    """Create a token and transform node."""
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
    return token, "node_test"
```
**Priority**: P1

### Gap: Missing FK constraint introspection tests
**Issue**: Tests assume FK constraints exist but never verify their presence in the schema
**Evidence**: All tests would silently pass if someone removed `ondelete="RESTRICT"` from schema.py
**Fix**: Add schema validation test at module level:
```python
def test_transform_errors_has_required_foreign_keys(landscape_db: LandscapeDB):
    """Verify transform_errors table has FK constraints on token_id and transform_id."""
    inspector = inspect(landscape_db._engine)
    fks = inspector.get_foreign_keys("transform_errors")

    fk_columns = {fk["constrained_columns"][0]: fk for fk in fks}

    assert "token_id" in fk_columns, "Missing FK constraint on token_id"
    assert fk_columns["token_id"]["referred_table"] == "tokens"
    assert fk_columns["token_id"]["options"].get("ondelete") == "RESTRICT"

    assert "transform_id" in fk_columns, "Missing FK constraint on transform_id"
    assert fk_columns["transform_id"]["referred_table"] == "nodes"
    assert fk_columns["transform_id"]["options"].get("ondelete") == "RESTRICT"

def test_validation_errors_has_required_foreign_keys(landscape_db: LandscapeDB):
    """Verify validation_errors table has nullable FK constraint on node_id."""
    inspector = inspect(landscape_db._engine)
    fks = inspector.get_foreign_keys("validation_errors")

    node_fk = next((fk for fk in fks if "node_id" in fk["constrained_columns"]), None)
    assert node_fk is not None, "Missing FK constraint on node_id"
    assert node_fk["referred_table"] == "nodes"
    assert node_fk["options"].get("ondelete") == "RESTRICT"
```
This prevents schema regressions where constraints are accidentally removed.
**Priority**: P1

### Gap: No test for FK violation error message clarity
**Issue**: Tests verify that IntegrityError is raised but don't verify the error message is actionable
**Evidence**: Regex `r"(FOREIGN KEY constraint failed|violates foreign key)"` is backend-specific but doesn't verify which FK failed
**Fix**: Add test to verify error messages contain enough context for debugging:
```python
def test_orphan_token_error_message_identifies_constraint(landscape_db: LandscapeDB, recorder: LandscapeRecorder):
    """FK violation error should identify which constraint failed for debugging."""
    run = recorder.begin_run(config={"test": True}, canonical_version="1.0")

    with pytest.raises(IntegrityError) as exc_info:
        with landscape_db.connection() as conn:
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_test",
                    run_id=run.run_id,
                    token_id="nonexistent_token",
                    transform_id="nonexistent_node",
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

    # Error should reference token_id or tokens table for actionable debugging
    error_msg = str(exc_info.value).lower()
    assert "token" in error_msg or "token_id" in error_msg, \
        "FK error should identify which constraint failed"
```
**Priority**: P3

### Gap: Missing parametrize for DB backends
**Issue**: FK constraint behavior varies between SQLite, PostgreSQL, MySQL - tests only run against one backend
**Evidence**: No `@pytest.mark.parametrize` for database backends, only runs with `LandscapeDB.in_memory()` (SQLite)
**Fix**: If ELSPETH supports multiple backends, add parametrized fixtures:
```python
@pytest.fixture(params=["sqlite", "postgresql"])
def landscape_db(request):
    if request.param == "sqlite":
        return LandscapeDB.in_memory()
    elif request.param == "postgresql":
        return LandscapeDB.from_url(os.getenv("TEST_POSTGRES_URL"))
```
This is only necessary if multi-backend support is a requirement. If SQLite-only, document this clearly.
**Priority**: P3 (or N/A if single-backend)

### Gap: No test for concurrent FK violations
**Issue**: FK checks can behave differently under concurrent access (race conditions)
**Evidence**: No tests verify FK constraints hold under concurrent inserts
**Fix**: Add test using threading:
```python
def test_concurrent_orphan_inserts_all_fail(landscape_db: LandscapeDB):
    """Concurrent attempts to insert orphan records should all fail atomically."""
    # This tests that FK checks are not vulnerable to TOCTOU races
    # Implementation would use threading.Thread to attempt parallel inserts
```
**Priority**: P3

### Gap: Missing test for NULL FK behavior in transform_errors
**Issue**: `transform_errors.token_id` is NOT NULL, but test suite doesn't explicitly verify NULL rejection
**Evidence**: Schema has `nullable=False` for token_id, but no test tries to insert NULL
**Fix**: Add test:
```python
def test_transform_errors_rejects_null_token_id(landscape_db: LandscapeDB, recorder: LandscapeRecorder):
    """transform_errors.token_id is NOT NULL - verify NULL is rejected."""
    run = recorder.begin_run(config={"test": True}, canonical_version="1.0")
    recorder.register_node(...)  # create valid transform node

    with pytest.raises(IntegrityError, match=r"(NOT NULL|null value)"):
        with landscape_db.connection() as conn:
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_null",
                    run_id=run.run_id,
                    token_id=None,  # Should be rejected
                    transform_id="node_test",
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()
```
This is defense-in-depth for schema correctness.
**Priority**: P2

## Positive Observations

1. **Excellent test naming**: Test names clearly state the constraint behavior being verified (`test_rejects_orphan_token_id`, `test_restrict_prevents_token_deletion`)

2. **Correct Tier 1 testing philosophy**: Tests directly assert that invalid data causes crashes (IntegrityError), aligning with "Our Data (Audit Database) - FULL TRUST" principle from CLAUDE.md

3. **Proper exception matching**: Uses `pytest.raises` with regex patterns to verify FK violations specifically, not generic exceptions

4. **NULL FK coverage**: `test_allows_null_node_id` correctly verifies nullable FK behavior for `validation_errors.node_id`, documenting the use case ("early validation failures")

5. **RESTRICT semantics coverage**: Tests verify both insert rejection (orphan records) and delete prevention (RESTRICT behavior)

6. **Direct SQLAlchemy Core usage**: Tests manipulate tables directly rather than going through high-level APIs, ensuring constraint behavior is tested at the database level

7. **Good comments**: Inline comments like `# ORPHAN - no such token` make test intent clear

8. **Proper transaction handling**: Uses context managers correctly with `landscape_db.connection()` and explicit `conn.commit()` in error paths

9. **Module docstring**: Clear explanation of what's being tested and links to the bug being addressed
