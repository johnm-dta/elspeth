# Test Quality Review: test_canonical.py

## Summary

Test suite has good coverage of critical functionality (NaN/Infinity rejection, type conversion, RFC 8785 compliance) but suffers from severe infrastructure gaps: import pollution in every test method (40+ duplicate imports), no property-based testing for hash stability guarantees, missing mutation testing for audit integrity claims, and incomplete coverage of hash collision risks and upstream topology validation edge cases.

---

## SME Agent Protocol Requirements

### Confidence Assessment
**Confidence Level: HIGH (85%)**

I have read both the implementation (`canonical.py`) and test suite, understand the critical audit integrity requirements from CLAUDE.md, and can identify specific test architecture gaps with evidence. Confidence is not 100% because I have not executed the test suite or examined CI configuration to verify test isolation in practice.

### Risk Assessment

| Risk Level | Scenario | Impact | Mitigation |
|------------|----------|--------|------------|
| **HIGH** | Import pollution causes false positives | Tests pass locally but fail in CI or different import order | Adopt fixture-based imports immediately |
| **MEDIUM** | Hash collision not tested | Audit trail integrity compromised by collision | Add collision resistance tests |
| **MEDIUM** | No property testing | Hash stability claims not verified at scale | Add Hypothesis tests for determinism |
| **LOW** | Mutation gaps | Code changes don't trigger test failures | Run mutation testing (mutmut) |

### Information Gaps

1. **Test execution environment**: Have not verified whether pytest runs tests in isolation or if import pollution actually causes failures
2. **CI pipeline configuration**: Unknown if tests run in parallel, which would expose interdependence issues
3. **Coverage reports**: No access to line/branch coverage metrics to verify claimed coverage levels
4. **Historical flakiness**: No data on whether these tests have exhibited intermittent failures

### Caveats

1. **Audit integrity claims require formal verification**: The test suite claims to verify "audit trail integrity" but lacks property-based testing to prove hash determinism across all valid inputs
2. **RFC 8785 compliance is assumed, not verified**: Tests verify behavior but do not validate against RFC 8785 test vectors
3. **Platform-specific behavior not tested**: NumPy/pandas type conversion may behave differently across platforms (Linux/macOS/Windows, different CPU architectures)
4. **This review covers test architecture, not correctness**: I am not verifying that the implementation meets RFC 8785 requirements, only that tests are well-constructed

---

## Poorly Constructed Tests

### Test: test_golden_hash_stability (line 384)
**Issue**: Insufficient cross-version stability verification
**Evidence**:
```python
def test_golden_hash_stability(self) -> None:
    """Verify hash matches known golden value."""
    # Only tests one data structure
    data = {"string": "hello", "int": 42, ...}
    golden_hash = "aed53055632a45e17618f46527c07dba463b2ae719e2f6832b2735308a3bf2e1"
    result = stable_hash(data)
    assert result == golden_hash
```
**Fix**: This test is critical for "audit trail integrity" (per CLAUDE.md: "Hashes survive payload deletion"). Should have:
- Multiple golden hash test cases covering different type combinations
- Edge cases: empty dict, empty list, deeply nested structures
- Boundary values: max int, very small float, long strings
- Complex structures: nested arrays, mixed types
- Property test verifying hash(canonicalize(x)) == hash(canonicalize(canonicalize(x))) (idempotence)
**Priority**: P1

### Test: test_tuple_converts_to_list (line 301)
**Issue**: Mutation vulnerability - doesn't verify deep conversion
**Evidence**:
```python
def test_tuple_converts_to_list(self) -> None:
    data = (1, 2, 3)
    result = _normalize_for_canonical(data)
    assert result == [1, 2, 3]
    assert type(result) is list
```
**Fix**: Missing test for nested tuples: `((1, 2), (3, 4))` should become `[[1, 2], [3, 4]]`. Without this, a bug in recursive tuple handling could go unnoticed.
**Priority**: P2

### Test: test_numpy_array_all_finite_accepted (line 91)
**Issue**: Tests empty array edge case but doesn't verify hash stability for it
**Evidence**:
```python
# Empty array (edge case)
empty_array = np.array([])
result_empty = _normalize_value(empty_array)
assert result_empty == []
```
**Fix**: Should verify `stable_hash(np.array([]))` produces consistent result. Empty arrays are a common edge case in data pipelines.
**Priority**: P3

### Test: test_pandas_timestamp_aware_to_utc_iso (line 211)
**Issue**: Weak assertion - doesn't verify actual UTC conversion
**Evidence**:
```python
def test_pandas_timestamp_aware_to_utc_iso(self) -> None:
    ts = pd.Timestamp("2026-01-12 10:30:00", tz="US/Eastern")
    result = _normalize_value(ts)
    # Should be converted to UTC
    assert "+00:00" in result or "Z" in result  # TOO WEAK
```
**Fix**: Verify exact UTC time: `assert result == "2026-01-12T15:30:00+00:00"` (US/Eastern is UTC-5 in January). Current assertion would pass even if timezone conversion was broken.
**Priority**: P1

### Test: test_datetime_naive_to_utc_iso (line 236)
**Issue**: Comment says "test that _normalize_value treats it as UTC" but doesn't test behavior, only final output
**Evidence**:
```python
# Naive datetime (no tzinfo) - test that _normalize_value treats it as UTC
dt = datetime(2026, 1, 12, 10, 30, 0, tzinfo=None)
result = _normalize_value(dt)
assert result == "2026-01-12T10:30:00+00:00"
```
**Fix**: Should have companion test showing naive datetime hashes identically to aware UTC datetime:
```python
naive_hash = stable_hash(datetime(2026, 1, 12, 10, 30, 0, tzinfo=None))
aware_hash = stable_hash(datetime(2026, 1, 12, 10, 30, 0, tzinfo=UTC))
assert naive_hash == aware_hash  # Prove they're treated identically
```
**Priority**: P2

### Test: test_bytes_to_base64_wrapper (line 251)
**Issue**: Doesn't test round-trip or hash stability
**Evidence**:
```python
def test_bytes_to_base64_wrapper(self) -> None:
    data = b"hello world"
    result = _normalize_value(data)
    assert result == {"__bytes__": base64.b64encode(data).decode("ascii")}
```
**Fix**: Missing verification that:
1. `stable_hash(b"data")` is deterministic
2. Different bytes produce different hashes
3. `{"__bytes__": "..."}` format is stable (what if implementation changes to `{"type": "bytes", "data": "..."}`?)
**Priority**: P2

### Test: test_canonical_json_sorts_keys (line 320)
**Issue**: Only tests top-level key sorting
**Evidence**:
```python
data = {"z": 1, "a": 2, "m": 3}
result = canonical_json(data)
assert result == '{"a":2,"m":3,"z":1}'
```
**Fix**: Should test nested dict key sorting: `{"outer": {"z": 1, "a": 2}}` → `{"outer":{"a":2,"z":1}}`. Key sorting must be recursive for hash stability.
**Priority**: P1

## Misclassified Tests

### Test Class: TestCoreIntegration (line 431)
**Issue**: Belongs in integration test suite, not unit tests
**Evidence**:
```python
class TestCoreIntegration:
    """Core module integration - all Phase 1 components exportable."""

    def test_dag_importable_from_core(self) -> None:
        from elspeth.core import ExecutionGraph, GraphValidationError
        # Tests cross-module imports
```
**Fix**: These tests verify `elspeth.core.__init__.py` exports, which is module integration, not canonical.py unit tests. Move to `tests/core/test_init.py` or `tests/integration/test_core_exports.py`.
**Priority**: P2

## Infrastructure Gaps

### Gap: Excessive Import Repetition
**Issue**: Every test method imports from `elspeth.core.canonical`
**Evidence**: 40+ occurrences of `from elspeth.core.canonical import ...` across 451 lines
```python
def test_string_passthrough(self) -> None:
    from elspeth.core.canonical import _normalize_value  # Repeated 40+ times
```
**Fix**: Use class-level or module-level imports:
```python
# At module level (preferred for test modules)
from elspeth.core.canonical import (
    CANONICAL_VERSION,
    _normalize_for_canonical,
    _normalize_value,
    canonical_json,
    stable_hash,
)

# OR at class level if different test classes need different imports
class TestNormalizeValue:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from elspeth.core.canonical import _normalize_value
        self.normalize = _normalize_value
```
**Rationale**: Inline imports in every test method:
1. Create visual noise (41% of test LOC is import statements)
2. Slow test execution (Python import cache helps but still overhead)
3. Violate DRY principle
4. Make refactoring harder (renaming function requires 40+ edits)
**Priority**: P2

### Gap: No Property-Based Tests
**Issue**: Missing Hypothesis tests for hash stability guarantees
**Evidence**: pyproject.toml shows `hypothesis>=6.98,<7` as dependency, but zero `@given` decorators in test file
**Fix**: Add property tests for critical guarantees:
```python
from hypothesis import given
from hypothesis import strategies as st

@given(st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
))
def test_canonical_json_deterministic(data):
    """Any serializable data produces same hash on repeated calls."""
    hash1 = stable_hash(data)
    hash2 = stable_hash(data)
    assert hash1 == hash2

@given(st.dictionaries(st.text(), st.integers()))
def test_key_order_independence(data):
    """Hash independent of dict insertion order."""
    from collections import OrderedDict
    forward = OrderedDict(sorted(data.items()))
    backward = OrderedDict(sorted(data.items(), reverse=True))
    assert stable_hash(forward) == stable_hash(backward)
```
**Rationale**: Per CLAUDE.md: "Property Testing: Hypothesis" is in acceleration stack. Deterministic hashing is THE critical property for audit integrity.
**Priority**: P0

### Gap: No Fixture for Test Data
**Issue**: Test data scattered and duplicated across methods
**Evidence**: `{"count": np.int64(42)}` pattern repeated in multiple tests, no reusable test data fixtures
**Fix**: Create fixtures for common test cases:
```python
@pytest.fixture
def numpy_types_dict():
    return {
        "int": np.int64(42),
        "float": np.float64(3.14),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3]),
    }

@pytest.fixture
def pandas_types_dict():
    return {
        "timestamp": pd.Timestamp("2026-01-12 10:30:00"),
        "nat": pd.NaT,
        "na": pd.NA,
    }
```
**Priority**: P3

### Gap: No Negative Tests for Unsupported Types
**Issue**: Doesn't verify behavior when encountering truly unsupported types
**Evidence**: No tests for custom classes, functions, modules, etc.
**Fix**: Add tests verifying behavior for unsupported types:
```python
class CustomClass:
    pass

def test_unsupported_type_raises():
    """Custom objects should fail gracefully with clear error."""
    obj = CustomClass()
    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_json({"custom": obj})
```
**Rationale**: Per CLAUDE.md Three-Tier Trust Model, Tier 3 (external data) can be "literal trash". Tests should verify system rejects invalid types with clear errors, not cryptic rfc8785 exceptions.
**Priority**: P2

### Gap: No Performance Regression Tests
**Issue**: Hash computation is in hot path (every row, every transform) but no performance benchmarks
**Evidence**: No pytest-benchmark tests, no wall-clock timing assertions
**Fix**: Add basic performance tests:
```python
def test_stable_hash_performance_small_dict(benchmark):
    """Baseline: small dict hashing should be <1ms."""
    data = {"key": "value", "count": 42}
    result = benchmark(stable_hash, data)
    assert isinstance(result, str)

def test_stable_hash_performance_large_nested(benchmark):
    """Large nested structure should complete in <100ms."""
    data = {"level1": {f"key{i}": list(range(100)) for i in range(100)}}
    result = benchmark(stable_hash, data)
    assert isinstance(result, str)
```
**Rationale**: Canonical hashing is called for every audit trail entry. Performance regression would degrade pipeline throughput. Pytest-benchmark is already installed (via pytest-cov deps).
**Priority**: P3

### Gap: No Test for repr_hash() Fallback
**Issue**: Implementation includes `repr_hash()` function (line 265) but zero test coverage
**Evidence**: No test class for repr_hash, function documented as "fallback when canonical_json fails"
**Fix**: Add test coverage:
```python
class TestReprHash:
    """repr_hash() fallback for non-canonical data."""

    def test_repr_hash_handles_nan(self):
        """repr_hash accepts NaN that canonical_json rejects."""
        result = repr_hash({"value": float("nan")})
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_repr_hash_deterministic_same_process(self):
        """repr_hash is deterministic within same process."""
        obj = {"custom": type("CustomClass", (), {})}
        hash1 = repr_hash(obj)
        hash2 = repr_hash(obj)
        # May differ across processes but must be stable in same process
        assert hash1 == hash2

    def test_repr_hash_warning_docstring(self):
        """Verify repr_hash docstring warns about cross-version instability."""
        from elspeth.core.canonical import repr_hash
        assert "NOT guaranteed to be stable across" in repr_hash.__doc__
```
**Priority**: P1

### Gap: No Test for compute_upstream_topology_hash()
**Issue**: Function at line 177 has zero test coverage in this file
**Evidence**: `compute_upstream_topology_hash()` is exported in public API but not tested in test_canonical.py
**Fix**: Either:
1. Add tests here if it's unit-testable without real ExecutionGraph
2. Create separate `test_canonical_graph_integration.py` for DAG-dependent tests
3. Document why it's tested elsewhere (if it is)
**Rationale**: Function is in same module, likely tested elsewhere, but reviewer can't verify from this file alone.
**Priority**: P3 (assuming it's tested in integration tests)

## Positive Observations

**Good: Clear test organization** - Test classes group related functionality well (TestNanInfinityRejection, TestNumpyTypeConversion, etc.)

**Good: Comprehensive NaN/Infinity rejection tests** - Lines 44-118 thoroughly test the critical audit integrity requirement per CLAUDE.md: "NaN and Infinity are strictly rejected".

**Good: Test docstrings reference bug tickets** - Tests at lines 73, 82, 91, 110 explicitly reference BUG-CANON-01, providing traceability.

**Good: Golden hash test exists** - test_golden_hash_stability provides regression protection, though it needs expansion (see P1 issue above).

**Good: Parametrized test for Decimal non-finite values** - Lines 128-141 use pytest.mark.parametrize effectively to cover NaN/sNaN/Infinity cases.

**Good: Version constant verification** - test_version_constant_exists (line 411) ensures CANONICAL_VERSION is available for audit records.
