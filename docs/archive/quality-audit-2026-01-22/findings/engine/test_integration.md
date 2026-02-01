# Test Quality Review: test_integration.py

## Summary

Integration test suite for the SDA engine shows strong fundamentals with comprehensive audit trail coverage, but suffers from severe infrastructure debt: 3500+ lines of test code with massive duplication, fixture gaps, and sleepy assertions. The test suite verifies critical contracts (audit spine, no silent loss) but repeats setup patterns 20+ times.

## Poorly Constructed Tests

### Test: test_missing_edge_error_is_not_catchable_silently (line 630)
**Issue**: Test verifies exception inheritance hierarchy but doesn't actually test catchability
**Evidence**:
```python
# Create an instance and verify attributes
error = MissingEdgeError(node_id="gate_1", label="nonexistent")
assert error.node_id == "gate_1"
assert error.label == "nonexistent"
assert "gate_1" in str(error)
assert "nonexistent" in str(error)
assert "Audit trail would be incomplete" in str(error)
```
The test name promises to verify the error "is not catchable silently" but it only checks attributes and string representations. It should actually attempt to catch the error with broad handlers to verify it surfaces correctly.
**Fix**: Add actual exception handling test:
```python
try:
    try:
        raise MissingEdgeError(node_id="gate_1", label="nonexistent")
    except Exception:
        pass  # Broad catch should NOT hide audit errors
    assert False, "MissingEdgeError was silently swallowed"
except MissingEdgeError:
    pass  # Correct - error propagated
```
**Priority**: P2

### Test: _build_test_graph function (line 37)
**Issue**: Critical helper function mutates internal graph state directly, bypassing public APIs
**Evidence**:
```python
# Populate internal ID maps
graph._sink_id_map = sink_ids
graph._transform_id_map = transform_ids
graph._config_gate_id_map = config_gate_ids
graph._route_resolution_map = route_resolution_map
graph._output_sink = output_sink
```
This directly assigns to private `_` prefixed attributes, violating encapsulation. If `ExecutionGraph` changes internals, 20+ tests break silently.
**Fix**: Create proper graph builder API or use `from_config()` when available. Comment at line 38 says "temporary until from_config is wired" but this tech debt persists across 3500 lines.
**Priority**: P0 - This is architectural debt that makes tests brittle

### Test: Multiple tests use hardcoded delays without explanation
**Issue**: No sleep calls found, but retry tests use arbitrary timing values
**Evidence**: Lines 2547-2549:
```python
retry={
    "max_attempts": 5,
    "initial_delay_seconds": 0.001,
    "max_delay_seconds": 0.01,
},
```
Good: Tests use fast retry delays for speed. However, no comment explains why 0.001s is safe or if this could cause flakiness on slow CI.
**Fix**: Add comment explaining timing safety margin
**Priority**: P3

### Test: test_empty_source_still_records_run (line 819)
**Issue**: Test verifies run/node recording but doesn't verify node_states are empty
**Evidence**: Test checks `len(nodes) == 3` but doesn't verify `get_node_states()` returns empty list, which is the actual audit requirement.
**Fix**: Add assertion:
```python
# Verify no node_states exist (no rows processed)
all_states = []
for node in nodes:
    states = recorder.get_node_states_for_node(node.node_id)
    all_states.extend(states)
assert len(all_states) == 0, "Empty source should have no node_states"
```
**Priority**: P2 - Incomplete verification of audit invariant

## Misclassified Tests

### Test: test_can_import_all_components (line 124)
**Issue**: Import test doesn't belong in integration suite
**Evidence**: Test only imports modules and checks `is not None`. This is a unit-level smoke test, not an integration test. Belongs in `tests/engine/test_init.py` or similar.
**Fix**: Move to `tests/engine/test_module_exports.py`
**Priority**: P2

### Test: test_base_aggregation_deleted (line 1196)
**Issue**: Deletion verification test doesn't belong in integration suite
**Evidence**:
```python
def test_base_aggregation_deleted(self) -> None:
    """BaseAggregation should be deleted (aggregation is structural)."""
    import elspeth.plugins.base as base
    assert not hasattr(base, "BaseAggregation"), "..."
```
This tests that a class doesn't exist. This is a migration test, not integration. Belongs in `tests/plugins/test_base_protocol.py`.
**Fix**: Move to unit tests for plugin base classes
**Priority**: P3

### Test: test_full_feature_pipeline_deleted (line 2181)
**Issue**: Another deletion verification masquerading as integration test
**Evidence**: Same pattern as above - verifies `BaseAggregation` doesn't exist
**Fix**: Consolidate with other deletion tests in migration suite
**Priority**: P3

## Infrastructure Gaps

### Gap: Massive fixture duplication across test classes
**Issue**: `ListSource`, `CollectSink`, `SelectiveFailTransform` are redefined 10+ times
**Evidence**:
- `ListSource` defined at lines 180, 296, 429, 565, 665, 748, 832, 1223, 2465, 2625, 3266, 3386
- `CollectSink` defined at lines 213, 335, 446, 582, 693, 856, 1375, 1673, 1988, 2267, 2508, 2668, 3307, 3427
- Each redefinition is nearly identical with minor variations

**Fix**: Create pytest fixtures in `conftest.py`:
```python
@pytest.fixture
def list_source_factory():
    """Factory for creating ListSource instances with custom data."""
    def _factory(data: list[dict[str, Any]], schema=ValueSchema):
        class _ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = schema
            def __init__(self):
                self._data = data
            def on_start(self, ctx): pass
            def load(self, ctx):
                for row in self._data:
                    yield SourceRow.valid(row)
            def close(self): pass
        return _ListSource()
    return _factory

@pytest.fixture
def collect_sink():
    """Reusable in-memory sink for integration tests."""
    # Similar pattern
```
**Priority**: P0 - 500+ lines of duplicated boilerplate

### Gap: No shared schema definitions
**Issue**: `ValueSchema` redefined 8+ times with identical structure
**Evidence**: Lines 175, 293, 427, 562, 662, 745, 829, 2220, 2462, 2622, 3263, 3383
```python
class ValueSchema(PluginSchema):
    value: int
```
**Fix**: Define once in `tests/engine/conftest.py`:
```python
class IntValueSchema(PluginSchema):
    """Standard schema for value: int tests."""
    value: int

class NumberSchema(PluginSchema):
    """Standard schema for n: int tests."""
    n: int
```
**Priority**: P1

### Gap: No test data builders
**Issue**: Test data created inline with magic numbers
**Evidence**: Line 2322:
```python
source = ListSource([{"value": i} for i in range(1, 11)])  # 1-10
```
Line 2691:
```python
source = ListSource([{"value": 1}, {"value": 2}])
```
**Fix**: Create builder functions:
```python
def build_int_sequence(start=1, count=10):
    """Build sequence of {value: int} dicts for testing."""
    return [{"value": i} for i in range(start, start + count)]

def build_selective_fail_data():
    """Build data where even values fail, odd succeed."""
    return build_int_sequence(0, 10)
```
**Priority**: P2

### Gap: No database fixture cleanup verification
**Issue**: Tests create in-memory databases but don't verify isolation
**Evidence**: Every test creates `LandscapeDB.in_memory()` but there's no verification that databases are actually isolated. If `in_memory()` reuses connections, tests could pollute each other.
**Fix**: Add fixture that verifies DB isolation:
```python
@pytest.fixture
def isolated_db():
    """Ensure each test gets a truly isolated database."""
    db = LandscapeDB.in_memory()
    yield db
    # Verify no leakage - could check connection pool stats
```
**Priority**: P2

### Gap: No shared PipelineConfig builder
**Issue**: `PipelineConfig` instantiation repeated with similar patterns
**Evidence**: Config created at lines 237, 360, 482, 614, 716, 879, 962, 1340, etc.
**Fix**: Create builder:
```python
def build_linear_pipeline(source, transforms, sink, gates=None):
    """Build simple linear pipeline: source -> transforms -> gates -> sink."""
    return PipelineConfig(
        source=as_source(source),
        transforms=transforms,
        sinks={"default": sink},
        gates=gates or [],
    )
```
**Priority**: P1

### Gap: Audit verification is ad-hoc and incomplete
**Issue**: Audit trail verification logic is repeated and inconsistent
**Evidence**: Compare line 373-409 (comprehensive audit spine check) vs line 892-898 (minimal check). Some tests verify full lineage, others only check run exists.
**Fix**: Create audit verification helpers:
```python
def verify_audit_spine_complete(recorder, run_id, expected_row_count):
    """Verify audit spine integrity for all rows."""
    rows = recorder.get_rows(run_id)
    assert len(rows) == expected_row_count

    for row in rows:
        tokens = recorder.get_tokens(row.row_id)
        assert len(tokens) >= 1, f"Row {row.row_id} has no tokens"

        for token in tokens:
            states = recorder.get_node_states_for_token(token.token_id)
            assert len(states) > 0, f"Token {token.token_id} has no states"
            # ... comprehensive checks
```
**Priority**: P0 - This is the core audit guarantee

### Gap: No property-based testing for invariants
**Issue**: Test suite manually creates edge cases but misses combinations
**Evidence**: Tests verify specific scenarios (2 rows, 3 rows, 10 rows) but don't test arbitrary N rows with random data. Project uses Hypothesis in stack but not in these tests.
**Fix**: Add property tests:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20))
def test_audit_spine_holds_for_any_input_size(values):
    """Audit spine must be complete regardless of input size."""
    source_data = [{"value": v} for v in values]
    # Run pipeline and verify audit spine
    # This would catch edge cases like N=1, N=large, etc.
```
**Priority**: P1 - Catches edge cases manual tests miss

### Gap: No explicit test for token identity uniqueness
**Issue**: Tests assume token IDs are unique but never verify
**Evidence**: Multiple tests check `len(tokens) == N` but don't verify `len(set(t.token_id for t in tokens)) == N`
**Fix**: Add invariant test:
```python
def test_token_ids_are_globally_unique():
    """Token IDs must be unique across entire run."""
    # Create pipeline with fork/coalesce to maximize token creation
    # Collect all tokens
    all_token_ids = [t.token_id for t in all_tokens]
    assert len(all_token_ids) == len(set(all_token_ids)), "Duplicate token IDs found"
```
**Priority**: P2

## Positive Observations

**Excellent audit spine coverage**: `test_audit_spine_intact` (line 276) and `test_audit_spine_with_routing` (line 411) are exemplary tests. They verify the core audit guarantee: every row → tokens → node_states → terminal state. This is exactly what integration tests should do.

**Good use of docstrings**: Tests explain WHAT they verify and WHY it matters (e.g., line 277: "proves chassis doesn't wobble").

**Comprehensive error path coverage**: Tests verify both success paths (line 163) and error paths (line 653, 736), including retry exhaustion (line 2597) and partial failures (line 3244).

**No sleepy assertions**: Tests use real synchronous execution, not time-based waits. Good.

**Proper distinction between FAILED and QUARANTINED**: `TestErrorRecovery` class correctly tests the two failure modes per project standards.

**Fork/coalesce integration is thorough**: `TestForkCoalescePipelineIntegration` (line 1208) verifies the complex diamond DAG pattern end-to-end.

**Metric verification is exhaustive**: `test_run_result_captures_all_metrics` (line 2193) checks all RunResult fields with consistency invariants.

## Recommendations

1. **IMMEDIATE (P0)**: Extract shared fixtures to `conftest.py` - this is 500+ lines of duplication
2. **IMMEDIATE (P0)**: Create `verify_audit_spine_complete()` helper - audit integrity is the core guarantee
3. **IMMEDIATE (P0)**: Fix `_build_test_graph` to not mutate private state
4. **HIGH (P1)**: Add property-based tests for audit invariants
5. **HIGH (P1)**: Create test data builders to eliminate magic numbers
6. **MEDIUM (P2)**: Move import tests to unit suite
7. **MEDIUM (P2)**: Verify DB isolation between tests
8. **LOW (P3)**: Consolidate deletion verification tests

## Test Count Summary

- Total test methods: ~35
- Properly constructed: ~30 (86%)
- Infrastructure issues: ALL tests affected by duplication
- Misclassified: 3 tests
- Lines of code: 3518
- Lines that could be eliminated with fixtures: ~500 (14%)
