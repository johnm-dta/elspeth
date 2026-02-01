# Test Quality Review: test_database.py

## Summary
The test file validates database initialization and schema compatibility but has significant gaps in testing Tier 1 audit integrity requirements. Missing: foreign key constraint validation, NOT NULL enforcement, unique constraint verification, and comprehensive schema compatibility tests covering all 19+ tables.

## Poorly Constructed Tests

### Test: test_connect_creates_tables (line 10)
**Issue**: Incomplete table verification - only checks 2 of 19+ tables
**Evidence**:
```python
assert "runs" in tables
assert "nodes" in tables
# Missing: edges, rows, tokens, token_parents, node_states, calls, artifacts,
# routing_events, batches, batch_members, batch_outputs, token_outcomes,
# validation_errors, transform_errors, checkpoints
```
**Fix**: Either verify all critical tables or rename test to indicate limited scope (e.g., `test_connect_creates_core_tables`). For database initialization tests, comprehensive table checking is expected.
**Priority**: P2

### Test: test_in_memory_factory (line 64)
**Issue**: Duplicate of test_connect_creates_tables with same incomplete coverage
**Evidence**: Only checks `"runs" in inspector.get_table_names()` - ignores 18+ other tables
**Fix**: Merge with test_connect_creates_tables or add value by checking different properties (indexes, constraints). Currently provides zero additional coverage.
**Priority**: P2

### Test: test_in_memory_enables_wal_mode (line 123)
**Issue**: Weak assertion accepts "memory" mode as equivalent to "wal"
**Evidence**:
```python
assert mode in ("wal", "memory"), f"Expected wal or memory, got {mode}"
```
**Fix**: Document that in-memory databases fundamentally cannot use WAL (no disk persistence), and this test verifies SQLite doesn't error when setting WAL on in-memory DBs, not that WAL is actually enabled. Alternatively, split into two tests: one for file-based WAL verification (strict), one for in-memory graceful handling (permissive).
**Priority**: P3

### Test: test_from_url_skip_table_creation (line 95)
**Issue**: Misleading test - creates empty database file but doesn't test the documented use case
**Evidence**:
```python
# Connect with create_tables=False - should NOT create tables
db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=False)
inspector = inspect(db.engine)
assert "runs" not in inspector.get_table_names()  # No tables!
```
**Fix**: The docstring says "Set to False when connecting to an existing database" but the test creates an EMPTY database. Add a test that connects to a pre-populated database with create_tables=False and verifies it doesn't DROP or ALTER existing tables. Current test is valid but incomplete.
**Priority**: P2

### Test: test_old_schema_missing_column_fails_validation (line 187)
**Issue**: Only tests one missing column (expand_group_id) from _REQUIRED_COLUMNS list
**Evidence**: Database module defines `_REQUIRED_COLUMNS = [("tokens", "expand_group_id")]` but this is clearly a list designed to hold multiple entries. Test doesn't verify behavior when multiple columns are missing or when future columns are added.
**Fix**: Add parameterized test or loop through _REQUIRED_COLUMNS to verify each entry. Add test case for multiple simultaneous missing columns to verify error message quality.
**Priority**: P1

### Test: test_error_message_includes_remediation (line 276)
**Issue**: Weak assertion - checks for "Delete" OR "delete" but doesn't verify the full remediation message
**Evidence**:
```python
assert "Delete" in error_msg or "delete" in error_msg
assert str(db_path) in error_msg or "sqlite" in error_msg
```
**Fix**: Match against the actual error format defined in database.py lines 154-159. Use regex or exact substring match to ensure the message provides ALL required information: (1) what's missing, (2) database path, (3) remediation steps. Current test would pass for "please delete something" which is not helpful.
**Priority**: P2

## Missing Critical Tests

### Missing: Foreign key constraint enforcement (P0)
**Issue**: No tests verify that FK constraints are ENABLED and ENFORCED at runtime
**Evidence**: Module enables foreign keys via PRAGMA (database.py line 86) but no tests verify constraints are actually enforced. Compare to test_error_table_foreign_keys.py which has comprehensive FK validation.
**Fix**: Add tests that:
1. Verify inserting orphan row (FK to non-existent parent) raises IntegrityError
2. Verify deleting parent with existing children raises IntegrityError
3. Test at least: runs→nodes, nodes→node_states, rows→tokens, tokens→node_states
**Why P0**: Tier 1 trust model requires "crash on any anomaly" - unenforced FKs allow garbage data in audit trail
**Priority**: P0

### Missing: NOT NULL constraint enforcement (P0)
**Issue**: No tests verify schema's NOT NULL constraints are enforced
**Evidence**: Schema defines many NOT NULL columns (runs.started_at, nodes.plugin_name, rows.source_data_hash, etc.) but test_database.py never validates these crash on NULL insertion
**Fix**: Add test that attempts to insert NULL into NOT NULL columns and verifies IntegrityError. Sample columns to test:
- runs.started_at
- runs.config_hash
- nodes.plugin_name
- rows.source_data_hash
- tokens.created_at
**Why P0**: "NULL where unexpected = crash" per Three-Tier Trust Model
**Priority**: P0

### Missing: Unique constraint enforcement (P0)
**Issue**: No tests verify unique constraints prevent duplicate entries
**Evidence**: Schema defines UniqueConstraints (edges: run_id+from_node_id+label, rows: run_id+row_index, etc.) but no validation tests
**Fix**: Add tests for critical unique constraints:
- edges (run_id, from_node_id, label) - can't have two edges with same label from one node
- rows (run_id, row_index) - can't have duplicate row indices in same run
- node_states (token_id, node_id, attempt) - can't record same attempt twice
**Why P0**: Duplicate audit records indicate data corruption or replay attacks
**Priority**: P0

### Missing: Required foreign key validation coverage (P1)
**Issue**: database.py defines _REQUIRED_FOREIGN_KEYS but no tests verify this validation logic works
**Evidence**:
```python
_REQUIRED_FOREIGN_KEYS: list[tuple[str, str, str]] = [
    ("validation_errors", "node_id", "nodes"),
    ("transform_errors", "token_id", "tokens"),
    ("transform_errors", "transform_id", "nodes"),
]
```
These are in schema.py (lines 312, 331-332) but test_database.py never validates _validate_schema() catches missing FKs.
**Fix**: Add test that creates old schema WITHOUT these foreign keys and verifies SchemaCompatibilityError is raised with specific FK name in message
**Priority**: P1

### Missing: Index existence verification (P1)
**Issue**: Schema defines 20+ indexes for query performance but no tests verify they're created
**Evidence**: schema.py lines 287-304 define critical indexes. If indexes are missing, queries will be slow but tests won't catch it.
**Fix**: Add test that inspects database and verifies critical indexes exist:
- ix_nodes_run_id
- ix_tokens_row_id
- ix_node_states_token
- ix_node_states_node
- ix_routing_events_state
**Priority**: P1

### Missing: Multiple database backend testing (P2)
**Issue**: Tests only use SQLite, but module supports PostgreSQL
**Evidence**: database.py line 66 has PostgreSQL-specific logic path: `if self.connection_string.startswith("sqlite")` but no tests exercise the "else" path
**Fix**: Add PostgreSQL test markers (e.g., `@pytest.mark.postgres`) that:
1. Skip if PostgreSQL not available
2. Connect to real PostgreSQL instance
3. Verify tables created correctly
4. Verify foreign keys work (PostgreSQL syntax differs from SQLite)
**Priority**: P2

### Missing: Schema version evolution testing (P2)
**Issue**: No tests verify behavior when _REQUIRED_COLUMNS list grows
**Evidence**: Code comment says "columns that have been added since initial schema" but no test validates what happens when we add a SECOND required column
**Fix**: Create test database with schema version N, add new column to _REQUIRED_COLUMNS, verify error message lists BOTH missing columns
**Priority**: P2

### Missing: Connection context manager transaction semantics (P1)
**Issue**: test_connection_context_manager (line 74) doesn't verify transaction behavior
**Evidence**:
```python
def test_connection_context_manager(self) -> None:
    db = LandscapeDB.in_memory()
    with db.connection() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
```
This validates connection works but doesn't test auto-commit on success or auto-rollback on exception.
**Fix**: Add tests:
1. Insert row, exit context normally → verify row committed
2. Insert row, raise exception → verify row NOT committed (rollback)
3. Multiple operations in transaction → verify atomicity
**Priority**: P1

### Missing: Alembic migration table detection (P3)
**Issue**: No test verifies behavior when database has alembic_version table
**Evidence**: _validate_schema() is for catching "stale local audit.db files" but Alembic-managed databases should bypass validation. No test verifies this.
**Fix**: Create database with alembic_version table, verify _validate_schema() doesn't raise errors for missing columns (Alembic will handle migration)
**Priority**: P3

## Misclassified Tests

### TestDatabaseConnection class (lines 7-59)
**Issue**: Mix of unit and integration tests without clear boundaries
**Evidence**:
- test_connect_creates_tables - Integration (file system, SQLAlchemy, SQLite)
- test_sqlite_wal_mode - Integration (PRAGMA execution)
- test_context_manager - Unit (just checks attribute exists)
**Fix**: Split into:
- TestDatabaseInitialization (integration tests requiring real DB)
- TestDatabaseAPI (unit tests for interface contracts)
**Priority**: P3

### test_fresh_database_passes_validation (line 170)
**Issue**: Classified as "schema compatibility" but actually tests "no false positives"
**Evidence**: Test creates new DB and expects no errors - this is a happy path test, not a compatibility test
**Fix**: Move to TestDatabaseConnection class and rename to test_new_database_initializes_successfully
**Priority**: P3

## Infrastructure Gaps

### Gap: No shared fixture for testing constraint violations
**Issue**: Tests repeatedly create run→node→row→token setup for FK tests
**Evidence**: Test file uses inline setup in each test (no fixtures). Compare to test_error_table_foreign_keys.py which also lacks setup fixtures but has even more duplication.
**Fix**: Add to tests/core/landscape/conftest.py:
```python
@pytest.fixture
def landscape_with_run(landscape_db: LandscapeDB, recorder: LandscapeRecorder):
    """Database with a single run created."""
    run = recorder.begin_run(config={}, canonical_version="1.0")
    return landscape_db, recorder, run

@pytest.fixture
def landscape_with_node(landscape_with_run):
    """Database with run and source node."""
    db, recorder, run = landscape_with_run
    node = recorder.register_node(...)
    return db, recorder, run, node
```
**Priority**: P2

### Gap: No fixture for creating "old schema" databases
**Issue**: Three tests (lines 187, 244, 276) duplicate logic to create old schema databases
**Evidence**: Each test manually creates old schema with `create_engine` + `text()` SQL
**Fix**: Add fixture:
```python
@pytest.fixture
def old_schema_db(tmp_path: Path) -> Path:
    """Create SQLite database with old schema (missing expand_group_id)."""
    db_path = tmp_path / "old_schema.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE tokens (token_id TEXT PRIMARY KEY, ...)"))
    engine.dispose()
    return db_path
```
**Priority**: P2

### Gap: No property-based testing for schema validation
**Issue**: Schema validation logic has complex branching (table exists? column exists? FK exists?) but only tested with hand-crafted examples
**Evidence**: test_old_schema_missing_column_fails_validation tests ONE specific case
**Fix**: Use Hypothesis to generate:
- Random table names (some valid, some invalid)
- Random column names (some present, some missing)
- Random FK definitions (some satisfied, some violated)
Verify error messages are always actionable.
**Priority**: P3

### Gap: No performance tests for schema validation
**Issue**: _validate_schema() inspects all tables and columns - no test verifies it completes quickly
**Evidence**: On a database with 10,000+ runs, validation might be slow
**Fix**: Add benchmark test with @pytest.mark.slow:
```python
def test_validation_completes_quickly_on_large_database(landscape_db):
    # Insert 1000 nodes, 10000 rows, 50000 tokens
    start = time.perf_counter()
    landscape_db._validate_schema()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, "Validation should complete in <500ms"
```
**Priority**: P3

## Positive Observations

- Schema compatibility tests catch real migration issues (expand_group_id)
- Error messages include remediation steps (good UX)
- Tests use inspector API correctly (vendor-neutral)
- Test class organization by feature is clear (TestDatabaseConnection, TestPhase3ADBMethods, TestSchemaCompatibility)
- Tests properly use tmp_path for file-based databases (no cleanup issues)
- in_memory() factory is tested (important for test performance)

## Recommendations Summary

**P0 (Block RC-1 unless present):**
1. Add foreign key constraint enforcement tests (3-5 critical FKs)
2. Add NOT NULL constraint enforcement tests (5-8 critical columns)
3. Add unique constraint enforcement tests (3-4 critical constraints)

**P1 (Fix before GA):**
1. Test _REQUIRED_FOREIGN_KEYS validation logic
2. Test connection context manager transaction semantics
3. Test index creation
4. Complete test_old_schema_missing_column_fails_validation for all required columns

**P2 (Quality improvement):**
1. Add shared fixtures for constraint testing
2. Improve table existence assertions (check all tables or be explicit about subset)
3. Add test for create_tables=False with pre-populated database
4. Test schema evolution (adding new required columns)

**P3 (Nice to have):**
1. Add PostgreSQL backend tests
2. Property-based testing for schema validation
3. Reorganize test classes by integration vs unit
4. Performance benchmarks for validation on large databases
