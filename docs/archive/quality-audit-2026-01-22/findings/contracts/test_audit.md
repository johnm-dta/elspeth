# Test Quality Review: test_audit.py

## Summary

Basic smoke tests that verify dataclass instantiation but miss critical contract validation. Tests are structurally sound but lack coverage of invariants, mutations, and edge cases mandated by the auditability standard. Zero tests for 6 dataclasses (NodeStatePending, TokenOutcome, NonCanonicalMetadata, ValidationErrorRecord, TransformErrorRecord, TypedDicts). Missing negative tests, property tests, and invariant enforcement validation.

## Poorly Constructed Tests

### Test: test_create_run_with_required_fields (line 37)
**Issue**: Assertion-free test for critical audit requirements
**Evidence**: Creates `Run` but never validates that `config_hash`, `settings_json`, or `canonical_version` are actually required. Missing validation that these fields cannot be empty strings or None.
**Fix**: Add negative tests verifying Pydantic/dataclass validation rejects invalid values. Test that `config_hash=""` raises ValidationError, that `canonical_version=None` fails construction.
**Priority**: P1 - Audit trail integrity depends on these fields being present and valid

### Test: test_run_status_must_be_enum (line 55)
**Issue**: Tests obvious Python behavior, not contract invariants
**Evidence**: Lines 66-69 verify that enum fields work like enums (`.value` accessor exists). This is testing Python's enum implementation, not ELSPETH's contracts.
**Fix**: Test enum boundary violations - verify `status="invalid_string"` raises TypeError, not that `status.value == "completed"` works.
**Priority**: P2 - Low-value test, replace with boundary testing

### Test: test_run_with_export_status (line 71)
**Issue**: Duplicate of previous test, no new assertions
**Evidence**: Creates Run with ExportStatus enum, asserts `run.export_status.value == "pending"` - same pattern as test_run_status_must_be_enum.
**Fix**: Delete test or repurpose to verify optional field behavior (can be None, transitions from None to PENDING are valid).
**Priority**: P2 - Test redundancy

### Test: test_frozen_dataclass_immutable (line 350)
**Issue**: Only ONE immutability test for entire suite
**Evidence**: Only `NodeStateOpen` tested for immutability. No tests for Run, Node, Edge, Row, Token, Call, Artifact, Batch, etc.
**Fix**: Use parametrized test covering ALL frozen dataclasses or property test with Hypothesis to verify frozen constraint universally.
**Priority**: P1 - Audit integrity requires immutability across all contracts

### Test: test_node_type_is_enum / test_determinism_is_enum (lines 109, 127)
**Issue**: Repetitive enum accessor tests with zero boundary validation
**Evidence**: Lines 123-125 test `node.node_type.value == "gate"` - redundant with line 123 `node.node_type == NodeType.GATE`. No test for invalid enum values.
**Fix**: Remove `.value` accessor tests, add boundary tests: `NodeType("invalid")` raises ValueError, that typos in config files fail fast.
**Priority**: P2 - Replace low-value tests with boundary tests

### Test: test_union_type_annotation (line 335)
**Issue**: Tests type checker behavior, not runtime contract
**Evidence**: Lines 338-348 assign `NodeStateOpen` to `NodeState` union type and assert `state is not None`. This validates type annotations work, not that discriminated union logic is correct.
**Fix**: Test discriminator field enforcement - verify cannot create `NodeStateCompleted` without `output_hash`, cannot create base `NodeState` directly, status field discriminates correctly.
**Priority**: P1 - Discriminated unions are critical for audit trail type safety

### Test: test_completed_state_requires_output (line 298)
**Issue**: Tests successful construction, not requirement enforcement
**Evidence**: Creates `NodeStateCompleted` with all required fields but never tests that omitting `output_hash` or `completed_at` raises error.
**Fix**: Add negative test: `NodeStateCompleted(..., output_hash=None)` must fail construction.
**Priority**: P0 - Critical audit requirement: completed states MUST have output hashes

### Test: test_failed_state_has_error_fields (line 316)
**Issue**: Only tests successful case, not error_json structure
**Evidence**: Sets `error_json='{"error": "something went wrong"}'` but never validates it's valid JSON, that malformed JSON is rejected, or that error_json can be None for transient failures.
**Fix**: Test error_json validation - malformed JSON rejected, None accepted, verify error structure is preserved byte-for-byte.
**Priority**: P1 - Error data is part of audit trail

### Test: test_artifact_type_is_string_not_enum (line 465)
**Issue**: Documents anti-pattern without testing safety constraints
**Evidence**: Lines 479-481 confirm `artifact_type` accepts "any string" but no test for empty string, whitespace-only, or excessively long values that could break storage.
**Fix**: Test edge cases - empty string, 10KB string, Unicode, control characters. If unconstrained strings are unsafe, add validation and test it.
**Priority**: P1 - Unbounded user input to audit trail

### Test: test_row_lineage_with_purged_payload (line 737)
**Issue**: Critical auditability test with insufficient assertions
**Evidence**: Tests payload degradation but doesn't verify that `source_data_hash` can still validate integrity when payload is restored from backup or that hash verification still works.
**Fix**: Test hash verification explicitly - verify hash matches source_data when available, verify hash is immutable even when payload purged, test hash collision detection.
**Priority**: P0 - Hash integrity is core auditability guarantee

### Test: test_row_lineage_hash_always_present (line 757)
**Issue**: Tests presence but not validity or immutability
**Evidence**: Lines 773 assert `lineage.source_data_hash == "required_hash_value"` but never tests that hash cannot be empty, cannot be modified, cannot be None, must be valid hex/base64 format.
**Fix**: Test hash format validation - reject empty string, reject "not-a-hash", reject wrong length, verify hash format matches canonical hash output.
**Priority**: P0 - Invalid hashes destroy audit integrity

## Misclassified Tests

### Suite Classification: Should be split into unit + property tests
**Issue**: All tests are unit tests but missing property-based tests for invariants
**Evidence**: No use of Hypothesis despite CLAUDE.md listing it as acceleration stack technology for "property testing". Contracts have universal invariants (immutability, enum boundaries, hash formats) that should be property-tested.
**Fix**: Create `test_audit_properties.py` using Hypothesis to test:
- All frozen dataclasses are immutable (generate instances, verify mutation fails)
- All hash fields have valid format (generate random strings, verify validation)
- All enum fields reject invalid values (generate arbitrary strings)
- All required fields cannot be None (generate instances with st.none())
**Priority**: P0 - Missing entire class of tests for critical properties

### Tests: All enum tests (lines 55, 71, 109, 127, 165, 391, 407, 504, 558)
**Issue**: Should be single parametrized test, not 9 separate methods
**Evidence**: Nine nearly-identical tests verifying enum fields work like enums.
**Fix**: Replace with pytest parametrized test covering all enum fields:
```python
@pytest.mark.parametrize("model_cls,field,enum_cls,valid_value", [
    (Run, "status", RunStatus, RunStatus.COMPLETED),
    (Node, "node_type", NodeType, NodeType.SOURCE),
    # ... all enum fields
])
def test_enum_fields_accept_valid_values(model_cls, field, enum_cls, valid_value):
    # Test construction with valid enum
    # Test rejection of invalid string
```
**Priority**: P1 - Test suite maintainability

## Infrastructure Gaps

### Gap: No shared fixtures for common test data
**Issue**: Every test constructs `datetime.now(UTC)` independently
**Evidence**: Line 39, 59, 92, 120, 138, 151, 174, 187, etc. - 40+ instances of `datetime.now(UTC)`.
**Fix**: Create pytest fixture:
```python
@pytest.fixture
def fixed_timestamp() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
```
Use fixed timestamp for deterministic tests, avoid time-based flakiness.
**Priority**: P2 - Test determinism and readability

### Gap: No fixtures for valid ID formats
**Issue**: Tests use arbitrary strings for IDs without validating format
**Evidence**: `run_id="run-123"`, `node_id="node-1"`, `token_id="tok-123"` - inconsistent formats suggest no ID validation.
**Fix**: Create fixtures for valid IDs, test that malformed IDs (empty, whitespace, wrong prefix) are rejected if validation exists, or add validation if missing.
**Priority**: P1 - ID format consistency for audit queries

### Gap: No hash validation infrastructure
**Issue**: Tests use placeholder hashes without format validation
**Evidence**: `config_hash="abc123"` (line 43), `source_data_hash="abc123"` (line 193), `upstream_topology_hash="a" * 64` (line 661) - inconsistent formats.
**Fix**: Create hash validation fixtures:
```python
@pytest.fixture
def valid_sha256_hash() -> str:
    return "a" * 64  # Valid SHA-256 hex string

@pytest.fixture
def invalid_hashes() -> list[str]:
    return ["", "short", "not-hex", "z" * 64]
```
Test that contracts validate hash format if applicable.
**Priority**: P0 - Hash format critical for integrity verification

### Gap: No negative test infrastructure
**Issue**: Zero tests verify that invalid construction fails
**Evidence**: No `pytest.raises` blocks except frozen dataclass test (line 365). No ValidationError tests, no TypeError tests for wrong types.
**Fix**: Create parametrized negative test fixture:
```python
@pytest.mark.parametrize("invalid_field,invalid_value,expected_error", [
    ("run_id", None, ValidationError),
    ("status", "invalid", TypeError),
    ("started_at", "not-a-datetime", ValidationError),
])
def test_run_rejects_invalid_fields(invalid_field, invalid_value, expected_error):
    with pytest.raises(expected_error):
        Run(**{invalid_field: invalid_value, **VALID_RUN_DEFAULTS})
```
**Priority**: P0 - Validation testing completely missing

### Gap: No relationship/referential integrity tests
**Issue**: Models have foreign key relationships but no constraint tests
**Evidence**: `Token.row_id` references `Row.row_id`, `NodeState.node_id` references `Node.node_id`, but no tests verify constraints.
**Fix**: If referential integrity is enforced at database level (not dataclass), document that these are schema tests only. If enforced in code, test it. Create integration tests for database constraints.
**Priority**: P1 - Audit trail consistency depends on referential integrity

### Gap: No serialization round-trip tests
**Issue**: Dataclasses will be serialized to/from database but no testing
**Evidence**: No tests for JSON serialization, no tests for database row conversion, no tests that datetime/enum serialization is reversible.
**Fix**: Test serialization round-trips:
```python
def test_run_json_serialization_roundtrip():
    original = Run(...)
    json_str = to_json(original)  # Whatever serialization is used
    deserialized = from_json(json_str, Run)
    assert deserialized == original
```
**Priority**: P1 - Serialization bugs will corrupt audit trail

### Gap: No tests for discriminated union exhaustiveness
**Issue**: NodeState union has 3+ variants but no test verifying all are handled
**Evidence**: Tests create individual variants (OPEN, COMPLETED, FAILED) but no test that pattern matching or isinstance checks cover all cases.
**Fix**: Test that discriminator field `status` has exactly expected variants, that adding new variant without updating handler code fails CI.
**Priority**: P1 - Missing variant handlers will corrupt audit trail

## Positive Observations

- **Frozen dataclass test exists** (line 350) - Only test checking immutability, proves pattern works
- **Enum usage is correct** - Tests consistently use enum values, not strings (even if testing is shallow)
- **Naming is clear** - Test names describe what's being tested (even if assertions are weak)
- **Discriminated union structure tested** - NodeStateOpen/Completed/Failed variants are instantiated correctly

## Summary Statistics

- **Total tests**: 54
- **Tests with no negative cases**: 53 (98%)
- **Tests verifying obvious Python behavior**: ~15 (28%)
- **Tests covering audit requirements**: ~5 (9%)
- **Duplicate/redundant tests**: ~10 (19%)
- **Property-based tests**: 0
- **Serialization tests**: 0
- **Relationship integrity tests**: 0

## Recommended Action Plan

1. **Immediate (P0)**: Add negative validation tests for required fields, hash formats, enum boundaries
2. **Immediate (P0)**: Add property tests using Hypothesis for immutability, format validation, enum constraints
3. **Before RC-2 (P1)**: Add serialization round-trip tests for all dataclasses
4. **Before RC-2 (P1)**: Consolidate enum tests into parametrized suite
5. **Before RC-2 (P1)**: Create shared fixtures for timestamps, IDs, hashes
6. **Nice-to-have (P2)**: Remove redundant tests, add integration tests for referential integrity

**Bottom line**: This suite tests Python's dataclass implementation, not ELSPETH's audit contracts. Rewrite with focus on invariants, boundaries, and failure modes.

---

## Additional Findings (2026-01-25 Comprehensive Review)

### Missing Contract Coverage (P0)

#### 1. NodeStatePending - Zero Test Coverage
**Issue**: Entire variant untested despite being part of the discriminated union
**Evidence**: Implementation exists (audit.py:147-171), used for async operations (batch submission, external calls with deferred results)
**Impact**: Async operations record PENDING states. Untested code path means audit trail integrity for async operations is unverified
**Fix Required**:
```python
def test_pending_state_has_literal_status():
    """NodeStatePending.status is Literal[PENDING]."""
    state = NodeStatePending(
        state_id="state-1", token_id="token-1", node_id="node-1",
        step_index=0, attempt=1, status=NodeStateStatus.PENDING,
        input_hash="abc123", started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC), duration_ms=100.0
    )
    assert state.status == NodeStateStatus.PENDING

def test_pending_state_has_timing_but_no_output():
    """NodeStatePending has completed_at and duration_ms, but no output_hash."""
    # Validates the async-operation contract
```

#### 2. TokenOutcome - Complete Missing Coverage
**Issue**: Zero tests for TokenOutcome dataclass despite being part of AUD-001 audit integrity
**Evidence**: Implementation exists (audit.py:498-521), critical for terminal state tracking
**Impact**: TokenOutcome is part of explicit terminal state recording. "I don't know what happened" is never acceptable - but we haven't tested the contract is usable
**Fix Required**:
```python
class TestTokenOutcome:
    def test_create_token_outcome_with_required_fields():
        """TokenOutcome captures terminal state determination."""
        outcome = TokenOutcome(
            outcome_id="out-1", run_id="run-1", token_id="tok-1",
            outcome=RowOutcome.COMPLETED, is_terminal=True,
            recorded_at=datetime.now(UTC)
        )
        assert outcome.outcome == RowOutcome.COMPLETED

    def test_token_outcome_is_terminal_matches_row_outcome():
        """is_terminal flag must match RowOutcome.is_terminal property."""
        # Verify BUFFERED has is_terminal=False, all others True
```

#### 3. NonCanonicalMetadata - Zero Coverage
**Issue**: Dataclass with factory method and dict conversion untested
**Evidence**: Implementation audit.py:411-477 includes `from_error()` factory and `to_dict()` conversion
**Impact**: NonCanonicalMetadata records why data couldn't be canonicalized (NaN, Infinity). Per CLAUDE.md canonical requirements, this is defense-in-depth for audit integrity
**Fix Required**:
```python
def test_non_canonical_metadata_to_dict_roundtrip():
    """to_dict() produces expected structure for audit storage."""
    meta = NonCanonicalMetadata(
        repr_value="{'x': nan}", type_name="dict",
        canonical_error="NaN not JSON serializable"
    )
    d = meta.to_dict()
    assert d["__repr__"] == "{'x': nan}"
    assert d["__type__"] == "dict"
    assert d["__canonical_error__"] == "NaN not JSON serializable"

def test_non_canonical_metadata_from_error_captures_context():
    """from_error() factory captures repr, type, and error message."""
    data = {"value": float("nan")}
    error = ValueError("NaN not allowed")
    meta = NonCanonicalMetadata.from_error(data, error)
    assert "nan" in meta.repr_value.lower()
    assert meta.type_name == "dict"
    assert "not allowed" in meta.canonical_error
```

#### 4. ValidationErrorRecord & TransformErrorRecord - Missing
**Issue**: Error records completely untested
**Evidence**: Implementations exist (audit.py:392-409, 479-496)
**Impact**: Error records answer "why did row 42 get quarantined?" If untested, we don't verify the contract supports explain() query requirements

### Discriminated Union Invariant Gaps (P0)

**Issue**: Tests verify successful construction but not invariant enforcement
**Current Coverage**: Tests create valid instances of each NodeState variant
**Missing**: Tests that attempt to violate variant invariants

**Required Tests**:
```python
# NodeStateOpen invariants
def test_open_state_cannot_have_output_hash():
    """NodeStateOpen must not have output_hash (not produced yet)."""
    # This should either:
    # 1. Fail at construction (if dataclass validates)
    # 2. Be prevented by type system (frozen dataclass doesn't include field)
    # 3. Be documented as "type-checker only" constraint

# NodeStateCompleted invariants
def test_completed_state_requires_all_timing_fields():
    """NodeStateCompleted must have output_hash, completed_at, duration_ms."""
    # Verify omitting any required field fails construction
    # This is CRITICAL for audit integrity - incomplete records are fraud

# NodeStateFailed invariants
def test_failed_state_can_have_output_hash():
    """NodeStateFailed may have output_hash (partial results before failure)."""
    # Verify failed states support both with/without output_hash cases
```

**Auditability Impact**: If variants don't enforce invariants, audit trail can contain garbage like "OPEN state with an output_hash" which violates the state machine contract.

### Checkpoint.created_at Contradiction (P0)

**Issue**: Test validates behavior that contradicts schema requirements
**Evidence**:
- Test (line 691): `created_at=None` passes construction
- Schema doc (audit.py:332): `created_at: datetime  # Required - schema enforces NOT NULL (Tier 1 audit data)`
- Implementation: Field type is `datetime` not `datetime | None`

**Impact**: This is a Tier 1 audit data contract violation. Per Three-Tier Trust Model: "Bad data in audit trail = crash immediately"

**Fix**: DELETE test_checkpoint_created_at_optional OR fix implementation to match schema. If created_at is NOT NULL in database, contract MUST enforce this.

### Hash Integrity Validation Missing (P0)

**Issue**: Hash fields are core auditability guarantee but untested for format/integrity
**Fields**: config_hash, source_data_hash, content_hash, input_hash, output_hash, response_hash, reason_hash, row_hash, upstream_topology_hash, checkpoint_node_config_hash
**Current Testing**: Tests assign arbitrary strings ("abc123", "a" * 64, "required_hash_value")
**Missing**: Format validation, length requirements, character set constraints

**Per CLAUDE.md**: "Hashes survive payload deletion - integrity is always verifiable"

**Required Tests**:
```python
def test_hash_fields_reject_empty_strings():
    """Hash fields should not accept empty strings."""
    # If hashes can be empty, collision detection is meaningless

def test_hash_fields_have_consistent_format():
    """All hash fields should use same format (SHA-256 hex = 64 chars)."""
    # Verify all hash fields follow canonical_json hash contract

def test_hash_immutability_after_payload_purge():
    """Hashes must remain verifiable after payload deletion."""
    # Create RowLineage with payload
    # Simulate purge (source_data=None, payload_available=False)
    # Verify hash is still present, valid, and matches if payload restored
```

### Enum Exhaustiveness Gaps (P1)

**Issue**: Spot-checks specific enum values without proving coverage of all values
**Example**:
- RunStatus has 4+ values (running, completed, failed, cancelled)
- Tests only create RUNNING and COMPLETED (lines 46, 64)
- No test verifies FAILED or CANCELLED can be persisted/retrieved

**Fix**: Parametrized exhaustiveness tests
```python
@pytest.mark.parametrize("status", list(RunStatus))
def test_run_accepts_all_valid_statuses(status: RunStatus):
    """Run construction succeeds for all RunStatus enum members."""
    run = Run(..., status=status)
    assert run.status == status
    assert isinstance(run.status, RunStatus)
```

**Impact**: Per Three-Tier Trust Model Tier 1, untested enum values could fail deserialization from database.

### Ordinal Uniqueness Constraints Untested (P2)

**Issue**: Ordinal fields tested for basic assignment but not uniqueness
**Fields**: BatchMember.ordinal, TokenParent.ordinal, RoutingEvent.ordinal
**Current Tests**: Verify ordinals can be assigned 0, 1, 2
**Missing**: Verify ordinals are unique within their scope (batch_id, token_id, routing_group_id)

**Auditability Impact**: If ordinals aren't unique, ordering is ambiguous. Per "No inference - if it's not recorded, it didn't happen" - duplicate ordinals mean we can't determine true order.

**Clarification Needed**: Is uniqueness enforced at:
- Contract level (construction fails)?
- DB level (INSERT fails with UNIQUE constraint)?
- Application level (recorder validates)?

### Timestamp Edge Cases Missing (P2)

**Issue**: Datetime fields only tested with `datetime.now(UTC)` - no edge cases
**Missing Coverage**:
- Timezone requirements (must be UTC?)
- Naive datetime rejection
- Microsecond precision preservation
- Far-future/far-past values (Y2038, Y10K)

**Per Auditability**: Timestamps without timezone info are ambiguous. "I don't know when this happened in absolute time" violates high-stakes accountability.

---

## Confidence Assessment

**Confidence Level**: HIGH
**Basis**: Direct code inspection of both test file and implementation contracts. Property test file examined for overlap.

**Information Gaps**:
1. Validation layer location unclear - are constraints enforced at dataclass construction, database schema, or repository layer?
2. Hash format specification not documented - assumed SHA-256 hex but not verified
3. Ordinal uniqueness enforcement strategy unclear

**Caveats**:
1. Review focuses on contract testing gaps. Integration/E2E coverage not assessed.
2. Database constraint enforcement assumed but not verified (would require Alembic migration review)
3. Some "missing" tests may be intentionally deferred to property test suite or integration tests

---

## Risk Assessment

**High Risk (P0)**:
- Untested NodeStatePending variant could corrupt async operation audit trail
- Checkpoint.created_at contradiction risks NULL in NOT NULL column
- Hash integrity validation missing risks invalid hashes in audit trail
- Discriminated union invariants unenforced risks type-unsafe audit records

**Medium Risk (P1)**:
- Enum exhaustiveness gaps could cause deserialization failures
- Missing error record tests risks explain() query failures
- Incomplete coverage of new dataclasses (TokenOutcome, NonCanonicalMetadata)

**Low Risk (P2-P3)**:
- Infrastructure gaps (fixtures, parametrization) - maintenance burden not correctness
- Timestamp edge cases - likely handled by database layer
- Ordinal uniqueness - likely enforced by database constraints

---

## Final Recommendations

### Before RC-1 Release (P0):
1. Add NodeStatePending test class (mirrors Open/Completed/Failed patterns)
2. Fix or delete Checkpoint.created_at contradiction test
3. Add hash format validation tests OR document that validation is DB-layer only
4. Add discriminated union invariant violation tests OR document as type-checker-only

### Before RC-2 (P1):
1. Add TokenOutcome, NonCanonicalMetadata, ValidationErrorRecord, TransformErrorRecord test classes
2. Add parametrized enum exhaustiveness tests
3. Add test fixtures for timestamps, IDs, hashes

### Maintenance Backlog (P2-P3):
1. Property-based tests for universal invariants (immutability, format validation)
2. Timestamp edge case coverage
3. Ordinal uniqueness clarification and testing
