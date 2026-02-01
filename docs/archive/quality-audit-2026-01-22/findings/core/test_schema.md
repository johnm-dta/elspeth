# Test Quality Review: test_schema.py

## Summary
Severely deficient test suite with minimal coverage, no constraint validation, poor isolation, and pervasive sleepy anti-patterns. Tests verify table existence but fail to validate the schema's core purpose: enforcing audit integrity through NOT NULL, foreign key, and unique constraints.

## Poorly Constructed Tests

### Test: test_create_all_tables (line 45)
**Issue**: Sleepy assertion - assumes table creation is instantaneous, no verification of constraint enforcement
**Evidence**:
```python
metadata.create_all(engine)
# Verify tables exist
inspector = inspect(engine)
tables = inspector.get_table_names()
assert "runs" in tables
```
Only checks table names exist. Doesn't verify columns, constraints, or indexes. A table with zero columns would pass this test.

**Fix**:
1. Verify column definitions match schema (names, types, nullable constraints)
2. Verify foreign key constraints exist and reference correct tables
3. Verify unique constraints exist
4. Verify indexes exist
5. Use inspector to get actual schema, compare against expected

**Priority**: P1

### Test: test_runs_table_exists (line 13)
**Issue**: Trivial existence check, no constraint validation
**Evidence**:
```python
assert runs_table.name == "runs"
assert "run_id" in [c.name for c in runs_table.columns]
```
Checks if `run_id` column exists but doesn't verify:
- `run_id` is primary key
- `started_at` is NOT NULL (audit requirement)
- `config_hash` is NOT NULL (audit requirement)
- `settings_json` is NOT NULL (audit requirement)
- `canonical_version` is NOT NULL (audit requirement)

**Fix**: Add comprehensive schema validation:
```python
def test_runs_table_schema():
    col = runs_table.c
    assert col.run_id.primary_key
    assert not col.started_at.nullable  # Tier 1 audit data
    assert not col.config_hash.nullable
    assert not col.settings_json.nullable
    assert not col.canonical_version.nullable
    assert not col.status.nullable
    assert col.completed_at.nullable  # Can be NULL while running
```

**Priority**: P0 - Violates CLAUDE.md Tier 1 trust requirement

### Test: test_nodes_table_exists through test_node_states_table_exists (lines 19-39)
**Issue**: Copy-paste pattern with zero schema validation
**Evidence**: All tests follow identical weak pattern:
```python
assert nodes_table.name == "nodes"
assert "node_id" in [c.name for c in nodes_table.columns]
```

These tables have critical constraints per schema.py:
- `nodes_table`: `run_id` FK NOT NULL, `plugin_name` NOT NULL, `determinism` NOT NULL
- `edges_table`: UniqueConstraint("run_id", "from_node_id", "label")
- `node_states_table`: UniqueConstraint("token_id", "node_id", "attempt")

**Fix**: Replace with constraint-validating tests for each table. Example for edges:
```python
def test_edges_table_foreign_keys():
    assert edges_table.c.run_id.foreign_keys  # Has FK to runs
    assert edges_table.c.from_node_id.foreign_keys  # Has FK to nodes
    assert edges_table.c.to_node_id.foreign_keys  # Has FK to nodes

def test_edges_table_unique_constraint():
    constraints = [c for c in edges_table.constraints if isinstance(c, UniqueConstraint)]
    # Verify exactly one unique constraint on (run_id, from_node_id, label)
    assert any(
        set(c.columns.keys()) == {"run_id", "from_node_id", "label"}
        for c in constraints
    )
```

**Priority**: P0 - Core audit integrity depends on these constraints

### Test: test_routing_event_model (line 123)
**Issue**: Intentional type error with `# type: ignore` comment hiding bug
**Evidence**:
```python
created_at=None,  # type: ignore[arg-type]  # Will be set in real use
```

Per CLAUDE.md: "Plugins are system-owned code. Plugin returns wrong type = CRASH - bug in our code"

**Fix**:
1. Remove `# type: ignore`
2. Use actual datetime: `created_at=datetime.now(UTC)`
3. If testing nullable behavior, that's a different test case entirely

**Priority**: P1 - Hides type bugs that should crash

### Test: test_batch_model (line 137)
**Issue**: Same type-ignore pattern, multiple instances
**Evidence**:
```python
created_at=None,  # type: ignore[arg-type]
```

This pattern appears in:
- test_routing_event_model (line 133)
- test_batch_model (line 147)
- test_node_model_has_determinism_field (line 176)
- test_checkpoint_model (line 248)
- test_checkpoint_model_with_aggregation_state (line 264)

**Fix**: Replace all with real datetime values. If testing NULL behavior, use integration test with database constraints.

**Priority**: P1 - Widespread type-safety violation

### Test: test_all_13_tables_exist (line 98)
**Issue**: Magic number with no assertion of schema stability
**Evidence**:
```python
def test_all_13_tables_exist(self) -> None:
    expected = {
        "runs", "nodes", "edges", "rows", "tokens",
        "token_parents", "node_states", "routing_events",
        "calls", "batches", "batch_members",
        "batch_outputs", "artifacts",
    }
```

Only 13 tables listed, but schema.py defines 17 tables:
- Missing: `token_outcomes`, `validation_errors`, `transform_errors`, `checkpoints`

**Fix**:
1. Update expected set to include all tables from schema.py
2. Use exact equality check, not subset:
```python
assert table_names == expected, f"Extra: {table_names - expected}, Missing: {expected - table_names}"
```

**Priority**: P0 - Test is factually wrong

## Misclassified Tests

### Test: TestPhase3AModels (entire class, lines 120-150)
**Issue**: Unit tests misclassified as schema tests
**Evidence**: Tests instantiate dataclass models directly:
```python
event = RoutingEvent(event_id="evt1", state_id="state1", ...)
assert event.event_id == "evt1"
```

This tests Python dataclass behavior, not database schema. Schema tests should:
1. Create actual database tables
2. Insert rows via SQLAlchemy
3. Verify constraints are enforced at DB level

**Fix**: Move to `tests/core/landscape/test_models.py`. Replace with integration tests that verify database rejects invalid data:
```python
def test_routing_events_foreign_key_enforced(test_db):
    with pytest.raises(IntegrityError):
        # Insert routing_event with non-existent state_id
        test_db.execute(insert(routing_events_table).values(
            event_id="e1", state_id="nonexistent", ...
        ))
```

**Priority**: P2 - Wrong layer, but not blocking

### Test: test_determinism_values (line 180)
**Issue**: Tests enum definition, not schema enforcement
**Evidence**:
```python
from elspeth.contracts import Determinism
valid_values = {d.value for d in Determinism}
```

This tests the enum class from contracts, not whether the database enforces valid values. SQLite doesn't enforce enum constraints unless using CHECK constraints or triggers.

**Fix**: Move to `tests/contracts/test_enums.py`. Add database constraint test:
```python
def test_nodes_table_rejects_invalid_determinism(test_db):
    # Attempt to insert node with invalid determinism value
    with pytest.raises(IntegrityError):  # or just succeeds if no CHECK constraint
        test_db.execute(insert(nodes_table).values(
            node_id="n1", run_id="r1", ..., determinism="invalid_value"
        ))
```

**Priority**: P3 - Misplaced but harmless

## Infrastructure Gaps

### Gap: No fixture for test database
**Issue**: Tests create engines inline, no cleanup, no reusable fixtures
**Evidence**: Lines 50-52:
```python
db_path = tmp_path / "test.db"
engine = create_engine(f"sqlite:///{db_path}")
metadata.create_all(engine)
```

Compare to `test_schema_not_null_constraints.py` which uses proper fixture:
```python
@pytest.fixture
def test_db(self, tmp_path: Path) -> LandscapeDB:
    db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
    return db
```

**Fix**: Add shared fixture in conftest.py or at module level:
```python
@pytest.fixture
def schema_test_db(tmp_path: Path) -> Engine:
    """Create test database with full schema."""
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    metadata.create_all(engine)
    yield engine
    engine.dispose()
```

**Priority**: P2 - Technical debt

### Gap: No constraint validation tests
**Issue**: ZERO tests verify database enforces constraints
**Evidence**: Grep for "IntegrityError" in test_schema.py: 0 results

Critical missing tests:
1. **Foreign key enforcement**: Insert with invalid FK should fail
2. **NOT NULL enforcement**: Insert NULL in non-nullable column should fail
3. **Unique constraint enforcement**: Insert duplicate should fail
4. **Primary key enforcement**: Insert duplicate PK should fail

The file `test_schema_not_null_constraints.py` exists and tests checkpoints table, proving the infrastructure works. `test_schema.py` should have similar tests for ALL tables.

**Fix**: Add comprehensive constraint validation tests. Example:
```python
def test_nodes_run_id_foreign_key(schema_test_db):
    """Verify nodes.run_id enforces foreign key to runs table."""
    with pytest.raises(IntegrityError):
        schema_test_db.execute(
            insert(nodes_table).values(
                node_id="n1",
                run_id="nonexistent_run",  # FK violation
                plugin_name="test",
                ...
            )
        )

def test_runs_started_at_not_null(schema_test_db):
    """Verify runs.started_at enforces NOT NULL constraint."""
    with pytest.raises(IntegrityError):
        schema_test_db.execute(
            insert(runs_table).values(
                run_id="r1",
                started_at=None,  # NOT NULL violation
                config_hash="hash",
                ...
            )
        )
```

**Priority**: P0 - CRITICAL GAP per CLAUDE.md Tier 1 requirements

### Gap: No index verification
**Issue**: Schema defines 15+ indexes (lines 286-303 in schema.py), zero tests verify they exist
**Evidence**: Lines 288-304 of schema.py:
```python
Index("ix_routing_events_state", routing_events_table.c.state_id)
Index("ix_routing_events_group", routing_events_table.c.routing_group_id)
# ... 13 more indexes
```

Indexes are performance-critical for audit queries. Missing indexes = O(n) table scans.

**Fix**: Add index verification test:
```python
def test_required_indexes_exist(schema_test_db):
    """Verify performance-critical indexes exist."""
    inspector = inspect(schema_test_db)

    # Check indexes for each table
    node_indexes = {idx['name'] for idx in inspector.get_indexes('nodes')}
    assert 'ix_nodes_run_id' in node_indexes

    routing_indexes = {idx['name'] for idx in inspector.get_indexes('routing_events')}
    assert 'ix_routing_events_state' in routing_indexes
    assert 'ix_routing_events_group' in routing_indexes
```

**Priority**: P1 - Performance regression risk

### Gap: No partial unique index verification
**Issue**: Token outcomes table has complex partial unique index (lines 141-149 in schema.py), zero tests
**Evidence**: Schema.py lines 141-149:
```python
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.is_terminal == 1),
    postgresql_where=(token_outcomes_table.c.is_terminal == 1),
)
```

This enforces "exactly one terminal outcome per token" - critical audit invariant. No test verifies this works.

**Fix**: Add integration test:
```python
def test_token_outcomes_one_terminal_per_token(test_db):
    """Verify partial unique index prevents multiple terminal outcomes."""
    # Insert first terminal outcome
    test_db.execute(insert(token_outcomes_table).values(
        outcome_id="o1", token_id="t1", outcome="completed",
        is_terminal=1, recorded_at=datetime.now(UTC)
    ))

    # Attempt second terminal outcome for same token - should fail
    with pytest.raises(IntegrityError):
        test_db.execute(insert(token_outcomes_table).values(
            outcome_id="o2", token_id="t1", outcome="failed",
            is_terminal=1, recorded_at=datetime.now(UTC)
        ))

    # Non-terminal outcomes can be duplicated (no constraint)
    test_db.execute(insert(token_outcomes_table).values(
        outcome_id="o3", token_id="t1", outcome="processing",
        is_terminal=0, recorded_at=datetime.now(UTC)
    ))  # Should succeed
```

**Priority**: P0 - Core audit correctness

### Gap: No multi-column unique constraint tests
**Issue**: Multiple tables have composite unique constraints, zero tests
**Evidence**: Schema.py defines:
- `edges_table`: UniqueConstraint("run_id", "from_node_id", "label") (line 84)
- `rows_table`: UniqueConstraint("run_id", "row_index") (line 99)
- `node_states_table`: UniqueConstraint("token_id", "node_id", "attempt") (line 186)
- `batch_members_table`: UniqueConstraint("batch_id", "ordinal") (line 272)

None are tested.

**Fix**: Add test for each unique constraint:
```python
def test_edges_unique_constraint(test_db):
    """Verify cannot create duplicate edge with same (run, from_node, label)."""
    # Insert first edge
    test_db.execute(insert(edges_table).values(
        edge_id="e1", run_id="r1", from_node_id="n1",
        to_node_id="n2", label="continue", default_mode="move",
        created_at=datetime.now(UTC)
    ))

    # Attempt duplicate (same run, from_node, label) - should fail
    with pytest.raises(IntegrityError):
        test_db.execute(insert(edges_table).values(
            edge_id="e2", run_id="r1", from_node_id="n1",
            to_node_id="n3", label="continue",  # Same label!
            default_mode="move", created_at=datetime.now(UTC)
        ))
```

**Priority**: P0 - Data integrity violations

## Missing Test Categories

### Category: Foreign Key Cascade Behavior
**Issue**: Schema uses `ondelete="RESTRICT"` for validation/transform errors (lines 312, 332), no tests verify this prevents deletion

**Fix**: Add test:
```python
def test_node_deletion_blocked_by_validation_errors(test_db):
    """Verify cannot delete node referenced by validation_errors (RESTRICT)."""
    # Create node and validation error referencing it
    test_db.execute(insert(nodes_table).values(...))
    test_db.execute(insert(validation_errors_table).values(node_id="n1", ...))

    # Attempt to delete node - should fail due to RESTRICT
    with pytest.raises(IntegrityError):
        test_db.execute(delete(nodes_table).where(nodes_table.c.node_id == "n1"))
```

**Priority**: P1 - Data safety

### Category: Column Type Verification
**Issue**: No tests verify column types match schema definition

**Fix**: Add comprehensive type check:
```python
def test_column_types_match_schema(schema_test_db):
    """Verify actual database column types match schema definition."""
    inspector = inspect(schema_test_db)

    runs_cols = {col['name']: col['type'] for col in inspector.get_columns('runs')}
    assert isinstance(runs_cols['run_id']['type'], String)
    assert runs_cols['run_id']['type'].length == 64
    assert isinstance(runs_cols['started_at']['type'], DateTime)
    assert runs_cols['started_at']['type'].timezone  # timezone=True
```

**Priority**: P2 - Regression prevention

### Category: Migration Compatibility
**Issue**: No tests verify schema.py matches Alembic migrations

The schema is defined in two places:
1. `schema.py` (SQLAlchemy Table definitions)
2. `alembic/versions/*.py` (migration scripts)

If they diverge, fresh installs get different schema than migrated databases.

**Fix**: Add test comparing `metadata.create_all()` output to running all migrations:
```python
def test_schema_matches_migrations(tmp_path):
    """Verify schema.py matches Alembic migrations."""
    # Create DB from schema.py
    engine1 = create_engine(f"sqlite:///{tmp_path}/schema.db")
    metadata.create_all(engine1)

    # Create DB from migrations
    engine2 = create_engine(f"sqlite:///{tmp_path}/migrated.db")
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", str(engine2.url))
    command.upgrade(alembic_cfg, "head")

    # Compare schemas
    inspector1 = inspect(engine1)
    inspector2 = inspect(engine2)

    tables1 = set(inspector1.get_table_names())
    tables2 = set(inspector2.get_table_names())
    assert tables1 == tables2, f"Table mismatch: {tables1 ^ tables2}"
```

**Priority**: P1 - Production correctness

## Positive Observations

1. **Test organization by phase**: Classes like `TestPhase3ASchemaAdditions` and `TestPhase5CheckpointSchema` clearly map to development phases
2. **Uses tmp_path fixture**: Proper isolation for database file creation (line 45)
3. **Imports are scoped**: Imports inside test methods avoid import-time side effects
4. **Type hints present**: All test methods have `-> None` return type annotations

## Recommended Actions

### Immediate (P0)
1. Fix `test_all_13_tables_exist` to include all 17 tables
2. Add NOT NULL constraint tests for all audit-critical fields (runs.started_at, runs.config_hash, nodes.run_id, etc.)
3. Add foreign key enforcement tests for at least runs/nodes/tokens hierarchy
4. Add unique constraint tests for edges, rows, node_states
5. Add partial unique index test for token_outcomes terminal state

### High Priority (P1)
1. Remove all `# type: ignore` comments, use real datetime values
2. Add index existence verification tests
3. Add migration compatibility test
4. Add foreign key cascade behavior tests
5. Verify constraint validation tests exist in proper integration test file

### Medium Priority (P2)
1. Extract shared test database fixture to conftest.py
2. Move model instantiation tests to test_models.py
3. Add column type verification tests
4. Add schema documentation verification (column comments, if supported)

### Low Priority (P3)
1. Move enum value tests to test_enums.py
2. Add property-based tests using Hypothesis for schema fuzzing
3. Add performance tests for indexed vs non-indexed queries

## Risk Assessment

**Current State**: Schema tests provide false confidence. They verify tables exist but not that constraints enforce audit integrity.

**Failure Mode**: Code could insert NULL into `runs.config_hash`, violating Tier 1 trust model. Test suite would pass. Audit trail corrupted.

**Blast Radius**: Every query depending on NOT NULL assumptions (every explain() query, every lineage trace) could fail with confusing errors or return wrong results.

**Pre-Release Criticality**: HIGH. Per CLAUDE.md: "WE ONLY HAVE ONE CHANCE TO FIX THINGS PRE-RELEASE." Schema bugs in production require data migrations.

## Compliance Check: CLAUDE.md Standards

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Tier 1 data must crash on anomaly | ❌ FAIL | No tests verify NOT NULL enforcement |
| No defensive patterns | ⚠️ PARTIAL | `# type: ignore` hides type bugs (lines 133, 147, 248, 264) |
| Schema as contract | ❌ FAIL | No constraint validation tests |
| Auditability standard | ❌ FAIL | Zero tests for audit-critical FK/NOT NULL constraints |
| No legacy code | ✅ PASS | No deprecated code patterns |
| Best practices, no compromises | ❌ FAIL | Existence checks instead of constraint validation |
