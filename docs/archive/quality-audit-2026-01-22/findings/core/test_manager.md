# Test Quality Review: test_manager.py

## Summary

The test suite for CheckpointManager is generally well-structured with good coverage of happy paths and edge cases. However, there are significant gaps in testing Tier 1 data integrity guarantees, test isolation issues due to fixture reuse, and missing property-based tests for critical operations. The suite over-relies on mutation testing to discover gaps instead of systematically verifying the "crash on anomaly" contract for audit data.

## Poorly Constructed Tests

### Test: test_create_checkpoint (line 92)
**Issue**: Weak assertions - only checks returned object, doesn't verify database persistence
**Evidence**:
```python
checkpoint = manager.create_checkpoint(...)
assert checkpoint.checkpoint_id is not None
assert checkpoint.run_id == "run-001"
assert checkpoint.sequence_number == 1
```
The test verifies the returned Checkpoint object but doesn't independently verify the database contains the checkpoint. A bug in the return statement could pass while the database insert silently failed.

**Fix**: Add explicit database verification after creation:
```python
# Verify checkpoint was actually persisted
loaded = manager.get_latest_checkpoint("run-001")
assert loaded is not None
assert loaded.checkpoint_id == checkpoint.checkpoint_id
assert loaded.upstream_topology_hash == checkpoint.upstream_topology_hash
```

**Priority**: P2

---

### Test: test_checkpoint_with_aggregation_state (line 123)
**Issue**: No verification of JSON serialization correctness - only round-trip equality
**Evidence**:
```python
assert json.loads(loaded.aggregation_state_json) == agg_state
```
This doesn't verify the JSON is actually valid or that specific types (numpy, pandas) are properly normalized per the canonical JSON requirements.

**Fix**: Add explicit JSON validation:
```python
# Verify JSON is valid (doesn't raise)
parsed = json.loads(loaded.aggregation_state_json)
# Verify no NaN/Infinity leaked through (Tier 1 contract)
assert parsed == agg_state
assert loaded.aggregation_state_json == '{"buffer":[1,2,3],"count":3}'  # Exact format
```

**Priority**: P2

---

### Test: test_old_checkpoint_rejected (line 220)
**Issue**: Directly inserts checkpoint instead of using create_checkpoint, violates test contract
**Evidence**: Lines 286-300 bypass CheckpointManager and insert directly into `checkpoints_table`. This is testing the validation logic but not the creation contract.

**Fix**: This is actually correct for testing the validation path in isolation. However, add a complementary test that verifies create_checkpoint itself cannot create an old checkpoint (by mocking datetime).

**Priority**: P3 (documentation issue - clarify why direct insert is used here)

---

### Test: test_new_checkpoint_accepted (line 312)
**Issue**: Directly inserts checkpoint, then only verifies created_at field comparison
**Evidence**:
```python
assert checkpoint.created_at.replace(tzinfo=None) == new_date.replace(tzinfo=None)
```
The test strips timezone information to compare, which masks the actual timezone handling behavior. This is Tier 1 data - we should crash if timezone is wrong, not coerce it.

**Fix**: Verify exact timezone preservation:
```python
# Tier 1 data - must preserve timezone exactly
if checkpoint.created_at.tzinfo is None:
    pytest.fail("created_at lost timezone information - Tier 1 data corruption")
assert checkpoint.created_at == new_date  # No coercion
```

**Priority**: P1 (violates Tier 1 "crash on anomaly" principle)

---

### Test: Missing test - hash collision detection
**Issue**: No test verifies behavior when two checkpoints get same checkpoint_id (UUID collision)
**Evidence**: Line 78 generates `checkpoint_id = f"cp-{uuid.uuid4().hex[:12]}"` with no collision handling

**Fix**: Add property-based test using Hypothesis to verify checkpoint_id uniqueness:
```python
@given(st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=100))
def test_checkpoint_ids_are_unique(self, manager, setup_run, mock_graph, sequence_numbers):
    """Creating multiple checkpoints must generate unique checkpoint_ids."""
    checkpoint_ids = set()
    for seq_num in sequence_numbers:
        cp = manager.create_checkpoint("run-001", "tok-001", "node-001", seq_num, mock_graph)
        assert cp.checkpoint_id not in checkpoint_ids, "Checkpoint ID collision detected"
        checkpoint_ids.add(cp.checkpoint_id)
```

**Priority**: P2 (low probability but catastrophic if it happens)

---

### Test: Missing test - transaction atomicity on failure
**Issue**: No test verifies that checkpoint creation is atomic (line 76: `with self._db.engine.begin()`)
**Evidence**: If topology hash computation fails (line 86-88), does the checkpoint row get rolled back?

**Fix**: Add test that triggers exception during checkpoint creation:
```python
def test_checkpoint_creation_rollback_on_hash_failure(self, manager, setup_run):
    """If hash computation fails, checkpoint must not be persisted (atomicity)."""
    # Create graph with node that will fail hash computation
    bad_graph = ExecutionGraph()
    bad_graph.add_node("node-001", node_type="transform", plugin_name="test", config={"bad": float('inf')})

    with pytest.raises(ValueError):  # canonical.py rejects NaN/Infinity
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1, bad_graph)

    # Verify no checkpoint was persisted (transaction rollback worked)
    assert manager.get_latest_checkpoint("run-001") is None
```

**Priority**: P0 (critical for audit integrity - partial writes are evidence tampering)

## Misclassified Tests

### Test: test_create_checkpoint_requires_graph (line 381)
**Issue**: Unit test masquerading as parameter validation test - should be in separate validation test class
**Evidence**: Tests are grouped in TestCheckpointManager but this is actually testing input validation, not checkpoint management behavior.

**Fix**: Move to separate test class `TestCheckpointManagerParameterValidation` with other validation tests (lines 381-430). This improves test organization and makes it clear these are defensive checks.

**Priority**: P3

---

### Test: test_checkpoint_with_empty_aggregation_state_preserved (line 141)
**Issue**: Regression test embedded in main test class without clear marker
**Evidence**: Docstring references Bug #6 but test has no `@pytest.mark.regression` or similar marker. Makes it hard to identify regression coverage.

**Fix**: Add explicit regression marker:
```python
@pytest.mark.regression(bug="P2-2026-01-19-checkpoint-empty-aggregation-state-dropped")
def test_checkpoint_with_empty_aggregation_state_preserved(self, ...):
```

**Priority**: P3

---

### Test: All checkpoint validation tests (lines 220-380)
**Issue**: Integration tests in unit test file - these test database interactions, not just CheckpointManager logic
**Evidence**: Tests require database setup via fixtures, insert data via SQLAlchemy, then test retrieval. This is integration testing (Manager + DB + Schema).

**Fix**: Either:
1. Move to `tests/integration/test_checkpoint_validation.py`, OR
2. Accept that CheckpointManager is inherently an integration point (I recommend this - it's a repository pattern)

If keeping as integration tests, add explicit marker:
```python
@pytest.mark.integration
class TestCheckpointManager:
```

**Priority**: P3 (classification issue, not correctness issue)

## Infrastructure Gaps

### Gap: Fixture duplication across test classes
**Issue**: `setup_run` fixture is duplicated in mutation gap tests and main test file
**Evidence**:
- Line 37-90 in test_manager.py
- Lines 45-97 in test_manager_mutation_gaps.py
- Lines 70-151 in conftest.py (different variant)

**Fix**: Consolidate into conftest.py with parametrization:
```python
@pytest.fixture
def setup_run(db: LandscapeDB, num_rows: int = 1) -> str:
    """Create a run with N tokens for checkpoint tests."""
    # Single implementation with configurable row count
```

**Priority**: P2 (maintenance burden - changes require updating 3 files)

---

### Gap: No shared assertion helpers for Tier 1 validation
**Issue**: Tests manually verify checkpoint fields but don't enforce "crash on anomaly" contract
**Evidence**: Line 346 does manual timezone comparison instead of using a shared validator:
```python
assert checkpoint.created_at.replace(tzinfo=None) == new_date.replace(tzinfo=None)
```

**Fix**: Create conftest helper for Tier 1 checkpoint validation:
```python
def assert_checkpoint_tier1_valid(checkpoint: Checkpoint) -> None:
    """Verify checkpoint satisfies Tier 1 data integrity contract.

    Tier 1 = OUR data. Crash on any anomaly.
    """
    assert checkpoint.checkpoint_id is not None, "checkpoint_id is NULL - Tier 1 violation"
    assert checkpoint.checkpoint_id.startswith("cp-"), f"Invalid checkpoint_id prefix: {checkpoint.checkpoint_id}"
    assert checkpoint.created_at.tzinfo is not None, "created_at missing timezone - Tier 1 violation"
    assert checkpoint.upstream_topology_hash is not None, "upstream_topology_hash is NULL - Bug #7"
    assert checkpoint.checkpoint_node_config_hash is not None, "checkpoint_node_config_hash is NULL - Bug #7"
    # Add more as needed
```

**Priority**: P1 (missing enforcement of core design principle)

---

### Gap: No test cleanup verification
**Issue**: Tests don't verify foreign key constraints are respected during delete
**Evidence**: `test_delete_checkpoints` (line 205) verifies checkpoint deletion but doesn't check if orphaned references exist

**Fix**: Add foreign key integrity tests:
```python
def test_delete_checkpoints_respects_foreign_keys(self, manager, setup_run):
    """Deleting run should cascade to checkpoints or prevent deletion."""
    # Create checkpoint
    manager.create_checkpoint(...)

    # Try to delete run (should cascade or raise FK error)
    # Verify behavior matches schema definition
```

**Priority**: P2

---

### Gap: No concurrency tests
**Issue**: CheckpointManager uses database transactions but no test verifies concurrent access
**Evidence**: Line 76 uses `with self._db.engine.begin()` which should provide isolation, but no test verifies two checkpoints created concurrently get unique sequence numbers

**Fix**: Add concurrency test using threading:
```python
def test_concurrent_checkpoint_creation(self, manager, setup_run, mock_graph):
    """Concurrent checkpoint creation must not cause sequence number conflicts."""
    import threading
    results = []

    def create_checkpoint(seq_num):
        cp = manager.create_checkpoint("run-001", "tok-001", "node-001", seq_num, mock_graph)
        results.append(cp)

    threads = [threading.Thread(target=create_checkpoint, args=(i,)) for i in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    # Verify all checkpoints created with unique checkpoint_ids
    checkpoint_ids = [cp.checkpoint_id for cp in results]
    assert len(checkpoint_ids) == len(set(checkpoint_ids))
```

**Priority**: P1 (production system will have concurrent access)

---

### Gap: Missing property-based tests for critical invariants
**Issue**: No Hypothesis tests to verify checkpoint invariants under randomized inputs
**Evidence**: Only example-based tests exist

**Fix**: Add property tests for:
1. Sequence number ordering (get_latest always returns max sequence)
2. Checkpoint ID uniqueness
3. Aggregation state round-trip (any valid JSON survives serialization)
4. Timezone preservation (any timezone-aware datetime preserves timezone)

**Priority**: P1 (Hypothesis recommended in CLAUDE.md Technology Stack)

---

### Gap: No test for database corruption detection
**Issue**: Tier 1 contract says "crash on anomaly" but no test verifies corrupt data causes crash
**Evidence**: No test manually corrupts checkpoint data (NULL in NOT NULL field, invalid enum) and verifies detection

**Fix**: Add corruption detection tests:
```python
def test_corrupted_checkpoint_id_crashes(self, manager, setup_run):
    """Reading checkpoint with NULL checkpoint_id must crash (Tier 1)."""
    from elspeth.core.landscape.schema import checkpoints_table

    # Manually insert corrupt checkpoint (bypass manager)
    with manager._db.engine.connect() as conn:
        conn.execute(checkpoints_table.insert().values(
            checkpoint_id=None,  # NULL in NOT NULL field
            run_id="run-001",
            # ... other fields
        ))

    # Reading must crash, not return None or coerce
    with pytest.raises(Exception):  # Should be specific exception type
        manager.get_latest_checkpoint("run-001")
```

**Priority**: P0 (Tier 1 integrity is non-negotiable)

## Positive Observations

1. **Good regression coverage**: Tests for Bug #6, #7, #9, #12 are clearly documented with bug references
2. **Edge case handling**: Empty aggregation state vs None distinction is well-tested (lines 141-193)
3. **Compatibility validation**: Old checkpoint rejection tests (lines 220-380) thoroughly cover the cutoff date logic
4. **Clear test organization**: Mutation gap tests are in separate file, making intent clear
5. **Realistic fixtures**: conftest.py provides shared fixtures with proper foreign key setup

## Additional Recommendations

### Missing Critical Test Cases

1. **Checkpoint with malformed aggregation JSON**: What if aggregation_state contains NaN/Infinity? (Should reject per canonical.py contract)
2. **get_checkpoints with no results**: Does empty list break ordering assertion?
3. **Sequence number overflow**: What happens with sequence_number = 2^63 - 1 (SQLite INTEGER limit)?
4. **Checkpoint node_id mismatch**: What if checkpoint references node_id not in current graph during recovery?
5. **Multiple runs isolation**: Creating checkpoint for run-001 shouldn't affect run-002's get_latest_checkpoint

### Test Organization Suggestions

1. Split TestCheckpointManager into logical groups:
   - TestCheckpointCreation
   - TestCheckpointRetrieval
   - TestCheckpointDeletion
   - TestCheckpointCompatibility
   - TestAggregationStatePersistence

2. Add pytest markers:
   - `@pytest.mark.tier1` for Tier 1 data integrity tests
   - `@pytest.mark.regression(bug="BUG-ID")` for regression tests
   - `@pytest.mark.property` for Hypothesis tests
   - `@pytest.mark.slow` for concurrency tests

3. Create `tests/core/checkpoint/test_manager_tier1.py` specifically for "crash on anomaly" tests

### Documentation Gaps

1. Test file missing module docstring explaining what aspects of CheckpointManager are tested
2. No explanation of why some tests directly insert to database (bypassing manager)
3. Missing cross-references to related test files (test_recovery.py, test_compatibility_validator.py)

## Risk Assessment

**Current Risk Level**: MODERATE

**Rationale**:
- Happy path coverage is good
- Mutation testing found format/ordering issues
- **Missing**: Tier 1 integrity enforcement, concurrency testing, property-based testing
- **Critical gap**: No verification that corrupt checkpoint data causes crashes instead of silent coercion

**Before RC-1 release**: Must add P0/P1 tests, especially:
1. Transaction atomicity on failure (P0)
2. Tier 1 corruption detection (P0)
3. Concurrency safety (P1)
4. Property-based invariants (P1)

**Standard**: "WE ONLY HAVE ONE CHANCE TO FIX THINGS PRE-RELEASE" - these gaps must be closed now, not post-release.
