# Test Quality Review: test_models.py

## Summary

`tests/core/landscape/test_models.py` contains 4 trivial smoke tests that merely verify dataclass instantiation. These tests provide **zero validation** that models enforce Tier 1 "crash on anomaly" requirements, zero validation of frozen dataclass immutability, zero type enforcement, and zero integration with the actual database schema they're meant to represent. For a subsystem that serves as the "legal record" and "source of truth" where "bad data = crash immediately", this is catastrophically inadequate.

**Overall Grade: F (Unacceptable for RC-1)**

---

## Critical Infrastructure Gaps

### Gap 1: No Database Integration Testing
**Issue**: Models are tested in isolation without validating they correctly map to SQLAlchemy schema
**Evidence**:
- Tests create model instances in memory only
- No database writes/reads to verify schema compatibility
- No validation that model fields match schema column types
- Compare to `test_schema.py` which creates actual database tables

**Impact**: P0
**Risk**: Models could accept values that violate database constraints (NOT NULL, foreign keys, unique constraints). Won't discover mismatch until production.

**Fix**: Add integration tests that:
```python
def test_run_persists_to_database(landscape_db):
    """Verify Run model serializes correctly to runs_table schema."""
    run = Run(...)
    landscape_db.insert_run(run)
    retrieved = landscape_db.get_run(run.run_id)
    assert retrieved.status == run.status  # Enum preserved
    assert retrieved.started_at == run.started_at  # Timezone preserved
```

### Gap 2: No Tier 1 Trust Enforcement Testing
**Issue**: No tests verify models crash on invalid data per Three-Tier Trust Model
**Evidence**: From CLAUDE.md:
> Tier 1: Our Data (Audit Database / Landscape) - FULL TRUST
> - Bad data in the audit trail = **crash immediately**
> - No coercion, no defaults, no silent recovery
> - Every field must be exactly what we expect - wrong type = crash, NULL where unexpected = crash

**Current tests**: Accept any values without validation. No tests for:
- NULL in required fields (should crash, not silently accept)
- Invalid enum strings (should crash, not coerce)
- Wrong datetime timezone (should crash on naive datetime)
- Invalid foreign key references

**Impact**: P0
**Risk**: Violates core audit integrity principle. If models accept garbage, audit trail becomes evidence tampering.

**Fix**: Add validation tests:
```python
def test_run_crashes_on_invalid_status_string():
    """Models must reject invalid enum values - no silent coercion."""
    with pytest.raises(ValueError):  # Or ValidationError if Pydantic
        Run(
            run_id="run-1",
            status="invalid_status",  # Not a valid RunStatus
            ...
        )

def test_run_crashes_on_naive_datetime():
    """All datetimes must be timezone-aware per audit standard."""
    with pytest.raises(ValueError):
        Run(
            run_id="run-1",
            started_at=datetime.now(),  # Missing UTC timezone
            ...
        )
```

### Gap 3: No Frozen Dataclass Immutability Testing
**Issue**: `NodeState` variants are marked `frozen=True` but never tested for immutability
**Evidence**: From models.py lines 115-225:
```python
@dataclass(frozen=True)
class NodeStateOpen:
    """Invariants: No output_hash, No completed_at, No duration_ms"""
```

**Current tests**: Zero validation that these are actually frozen or that invariants hold.

**Impact**: P1
**Risk**: Accidental mutation of audit records destroys traceability.

**Fix**:
```python
def test_node_state_open_is_immutable():
    """Frozen dataclasses must reject field mutation."""
    state = NodeStateOpen(...)
    with pytest.raises(FrozenInstanceError):
        state.status = NodeStateStatus.COMPLETED  # type: ignore
```

### Gap 4: Missing Discriminated Union Validation
**Issue**: `NodeState` is a discriminated union but no tests verify discriminator behavior
**Evidence**: From models.py line 228:
```python
NodeState = NodeStateOpen | NodeStatePending | NodeStateCompleted | NodeStateFailed
"""Use isinstance() or check the status field to discriminate"""
```

**Current tests**: Don't verify type narrowing or runtime discrimination works.

**Impact**: P2
**Risk**: Code relying on discriminator might fail at runtime if not properly enforced.

**Fix**:
```python
def test_node_state_discrimination_by_status():
    """Type checker should narrow based on status field."""
    state: NodeState = NodeStateOpen(
        state_id="s1",
        status=NodeStateStatus.OPEN,
        ...
    )
    if state.status == NodeStateStatus.OPEN:
        # Type checker should know this is NodeStateOpen
        assert not hasattr(state, 'output_hash')

def test_node_state_completed_invariants():
    """NodeStateCompleted MUST have output_hash, completed_at, duration_ms."""
    state = NodeStateCompleted(
        state_id="s1",
        status=NodeStateStatus.COMPLETED,
        output_hash="abc123",
        completed_at=datetime.now(UTC),
        duration_ms=42.0,
        ...
    )
    # Verify invariants
    assert state.output_hash is not None
    assert state.completed_at is not None
    assert state.duration_ms is not None
```

---

## Poorly Constructed Tests

### Test: test_create_run (line 10)
**Issue**: Trivial smoke test - only verifies dataclass constructor works
**Evidence**:
```python
def test_create_run(self) -> None:
    run = Run(run_id="run-001", ...)
    assert run.run_id == "run-001"
    assert run.status == RunStatus.RUNNING
```
This is **kindergarten-level testing**. It verifies Python's dataclass mechanism works, not that our model enforces audit requirements.

**Missing**:
- Validation that required fields crash when missing
- Validation that enums reject invalid strings
- Validation that datetimes must be timezone-aware
- Validation that JSON fields are actually valid JSON
- Hash field format validation (should match canonical hash format)

**Fix**: Replace with property-based testing:
```python
@given(
    run_id=st.text(min_size=1, max_size=64),
    config_hash=st.text(min_size=64, max_size=64),  # SHA-256
    settings_json=st.from_regex(r'\{.*\}'),  # Valid JSON object
)
def test_run_accepts_valid_inputs(run_id, config_hash, settings_json):
    """Run should accept any valid inputs per schema."""
    run = Run(
        run_id=run_id,
        started_at=datetime.now(UTC),
        config_hash=config_hash,
        settings_json=settings_json,
        canonical_version="sha256-rfc8785-v1",
        status=RunStatus.RUNNING,
    )
    assert run.run_id == run_id
```

**Priority**: P1

### Test: test_create_node (line 29)
**Issue**: Same trivial pattern - verifies dataclass works, not model correctness
**Evidence**: Only checks field assignment, not validation or constraints

**Missing**:
- Foreign key semantics (node.run_id should reference valid run)
- Plugin version format validation
- Config hash length validation (should be SHA-256 = 64 chars)
- Determinism enum validation
- Schema hash format when present

**Priority**: P1

### Test: test_create_row (line 51)
**Issue**: Doesn't validate critical audit trail requirement - row_index uniqueness within run
**Evidence**: From schema.py line 99:
```python
UniqueConstraint("run_id", "row_index"),
```

**Missing**:
- Validation that source_data_hash is valid hash format
- Validation that source_node_id references actual source node
- Row index is non-negative
- Created_at is timezone-aware

**Priority**: P1

### Test: test_create_token (line 68)
**Issue**: Trivial test, missing critical DAG lineage validation
**Evidence**: Token model has complex lineage fields (fork_group_id, join_group_id, expand_group_id, branch_name) - none tested

**Missing**:
- Token must reference valid row_id (foreign key)
- Fork/join/expand group IDs should be mutually exclusive or follow specific patterns
- Branch name validation when forked
- Parent token relationships (TokenParent table)

**Priority**: P2

---

## Misclassified Tests

### Test Class: TestRunModel, TestNodeModel, etc.
**Issue**: Labeled as "model" tests but actually "constructor smoke tests"
**Evidence**: No validation logic, no database interaction, no constraint testing

**Should be**: Unit tests for model validation logic OR integration tests for database mapping

**Current classification**: Somewhere between "not real tests" and "misleadingly named"

**Fix**: Either:
1. Rename to `TestDataclassInstantiation` (honest but useless)
2. Replace with real model tests (validation + database round-trip)

**Priority**: P2

---

## Missing Critical Test Coverage

### Missing: Checkpoint Model Topology Validation (Bug #7 Fix)
**Issue**: Checkpoint model has critical topology validation fields added for Bug #7 - zero tests
**Evidence**: From models.py lines 340-342:
```python
# Topology validation fields (Bug #7 fix - required for checkpoint validation)
upstream_topology_hash: str  # Hash of nodes + edges upstream of checkpoint
checkpoint_node_config_hash: str  # Hash of checkpoint node config only
```

**From schema.py line 357**:
```python
Column("upstream_topology_hash", String(64), nullable=False),
Column("checkpoint_node_config_hash", String(64), nullable=False),
```

**Missing tests**:
- These fields are NOT NULL - test crashes when missing
- Hash format validation (64 chars, hex)
- Semantic validation (topology hash matches actual upstream graph)

**Priority**: P0 (This was a bug fix - should have tests)

### Missing: RowLineage Graceful Degradation
**Issue**: RowLineage model has special payload purge semantics - zero tests
**Evidence**: From models.py lines 348-369:
```python
@dataclass
class RowLineage:
    """Lineage information for a row with graceful payload degradation.

    Used by explain_row() to report row lineage even when payloads
    have been purged. The hash is always preserved, but the actual
    data may be unavailable.
    """
    source_data: dict[str, object] | None
    """Original source data, or None if payload was purged."""

    payload_available: bool
    """True if source_data is available, False if purged or unavailable."""
```

**Missing tests**:
- Validation that source_hash is ALWAYS present (even when payload purged)
- Validation that payload_available correctly reflects source_data state
- Edge case: payload_available=True but source_data=None (should crash)
- Edge case: payload_available=False but source_data present (inconsistent state)

**Priority**: P1

### Missing: Export Status Independence Testing
**Issue**: Run model has export status separate from run status - no tests verify independence
**Evidence**: From schema.py lines 42-48:
```python
# Export tracking - separate from run status so export failures
# don't mask successful pipeline completion
Column("export_status", String(32)),  # pending, completed, failed, None
Column("export_error", Text),
```

**Missing tests**:
- Run can be COMPLETED with export_status=FAILED (export failure doesn't fail run)
- Run can be COMPLETED with export_status=None (export not configured)
- Export error message present only when export_status=FAILED

**Priority**: P2

### Missing: NodeState Invariant Testing
**Issue**: Each NodeState variant documents invariants in docstrings - zero tests verify them
**Evidence**: From models.py:

**NodeStateOpen invariants (lines 122-128)**:
- No output_hash
- No completed_at
- No duration_ms
- No error_json
- No context_after_json

**NodeStatePending invariants (lines 149-153)**:
- No output_hash (result not available yet)
- Has completed_at
- Has duration_ms
- No error_json

**NodeStateCompleted invariants (lines 176-180)**:
- Has output_hash
- Has completed_at
- Has duration_ms
- No error_json

**NodeStateFailed invariants (lines 204-208)**:
- Has completed_at
- Has duration_ms
- May have error_json
- May have output_hash

**Missing tests**: ZERO tests verify these invariants hold when constructing states.

**Priority**: P0 (These are documented contracts - must be enforced)

---

## Comparison to Adjacent Test Files

### test_models_enums.py (Good Practice)
**What it does well**:
- Explicitly tests enum type acceptance
- Clear test names describing what's validated
- Covers all model fields that use enums

**What test_models.py should learn**: Test explicit requirements, not just "does it instantiate"

### test_models_mutation_gaps.py (Excellent Practice)
**What it does well**:
- Uses fixtures for common model creation
- Tests each optional field defaults to None (not empty string, not 0)
- Tests type requirements (RunStatus enum, not string)
- Explicitly targets mutation testing gaps
- Clear docstrings explaining WHICH line is being tested

**Example**:
```python
def test_completed_at_defaults_to_none(self, minimal_run: Run) -> None:
    """Line 39: completed_at must default to None, not empty string or 0."""
    assert minimal_run.completed_at is None
    assert not isinstance(minimal_run.completed_at, str)
```

**What test_models.py should learn**: Be explicit about what you're testing and why.

---

## Positive Observations

**Type hints are present**: All tests use `-> None` return type annotation.

**Enum imports are correct**: Tests import enums from contracts package.

**UTC timezone usage**: Tests use `datetime.now(UTC)` consistently.

That's where the good news ends.

---

## Recommended Actions

### Immediate (P0)

1. **Add Tier 1 validation tests** - Models must crash on invalid data per audit standard
2. **Add database integration tests** - Round-trip to SQLAlchemy schema
3. **Add NodeState invariant tests** - Verify frozen dataclass contracts hold
4. **Add Checkpoint topology hash tests** - Validate Bug #7 fix fields

### Short-term (P1)

5. **Add foreign key semantic tests** - Models reference valid parent records
6. **Add RowLineage payload degradation tests** - Verify graceful purge behavior
7. **Replace trivial smoke tests** with property-based validation tests
8. **Add timezone-aware datetime enforcement** - Crash on naive datetimes

### Medium-term (P2)

9. **Add discriminated union tests** - Verify NodeState type narrowing
10. **Add export status independence tests** - Verify export doesn't affect run status
11. **Rename test classes** to reflect actual testing scope
12. **Consider mutation testing** - Current tests would have terrible mutation score

---

## Architecture-Level Issue

**Fundamental problem**: These models are **plain dataclasses** with no validation logic. For a Tier 1 "crash on anomaly" subsystem, this is architecturally wrong.

**Evidence from CLAUDE.md**:
> Reading from Landscape tables? Crash on any anomaly.

**But dataclasses don't validate**. They accept whatever you pass.

**Architectural fix needed**:
1. Use **Pydantic dataclasses** for validation at construction time
2. Or add explicit `__post_init__` validation to each model
3. Or use SQLAlchemy ORM with validators (but CLAUDE.md says "SQLAlchemy Core")

**Current state**: Models are "dumb bags of fields" that trust callers to provide valid data. This contradicts the Three-Tier Trust Model which says Landscape data is Tier 1 - full trust AFTER validation, not before.

**Who validates?** If not the models, then Recorder must validate before insert. But then why have models at all - just use dicts?

**This is an architectural defect that must be fixed properly per CLAUDE.md**: "WE ONLY HAVE ONE CHANCE TO FIX THINGS PRE-RELEASE. Make the fix right, not quick."

---

## Test Classification Recommendation

**Current**: Unit tests (but barely)
**Should be**: Hybrid approach

1. **Unit tests** for model validation logic (after adding validation)
2. **Integration tests** for database schema mapping
3. **Property tests** for invariant enforcement

**Test file should probably be split**:
- `test_models_validation.py` - Unit tests for model validators
- `test_models_database.py` - Integration tests for schema round-trip
- `test_models_invariants.py` - Property tests for NodeState contracts

---

## Quality Metrics

| Metric | Score | Target | Gap |
|--------|-------|--------|-----|
| Lines of test code | 77 | 500+ | -85% |
| Models tested | 4/24 | 24/24 | -83% |
| Database integration | 0% | 100% | -100% |
| Validation coverage | 0% | 100% | -100% |
| Invariant testing | 0% | 100% | -100% |
| Mutation score (estimated) | <10% | >80% | -70%+ |

**Conclusion**: For the "audit backbone" and "source of truth" subsystem of a high-stakes accountability system at RC-1, this test coverage is **unacceptably inadequate**. The tests provide a false sense of security while validating almost nothing of importance.
