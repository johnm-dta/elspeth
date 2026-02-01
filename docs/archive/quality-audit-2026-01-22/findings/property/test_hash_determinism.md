# Test Quality Review: test_hash_determinism.py

## Summary

The property tests exhibit good generator coverage and foundational property verification, but suffer from critical gaps in shrinking validation, NaN/Infinity rejection testing, mutation vulnerability defense, and cross-boundary determinism verification. Several tests exhibit redundancy with unit tests and fail to leverage property testing's strength in detecting edge cases through better generators.

## Poorly Constructed Tests

### Test: test_canonical_json_is_valid_json (line 97)
**Issue**: Assertion is a tautology that provides zero value
**Evidence**: `assert parsed is not None or data is None` always passes regardless of JSON validity. If `json.loads()` succeeds, `parsed` exists; if data is `None`, second clause is true. The assertion never fails.
**Fix**: Replace with meaningful validation:
```python
# Assert that round-trip produces structurally equivalent data
# (types may change: Python tuples → JSON arrays → Python lists)
if data is None:
    assert parsed is None
elif isinstance(data, (dict, list)):
    # For containers, verify structure preservation
    assert type(parsed) == type(data)
```
**Priority**: P1

### Test: test_canonical_json_no_whitespace (line 342)
**Issue**: Test verifies nothing meaningful due to passive `pass` statement
**Evidence**: Lines 352-355 contain logic to check for invalid spacing patterns, but end with `pass` and comment "Complex to verify without parsing - rely on rfc8785". Test claims to verify "no unnecessary whitespace" but explicitly skips verification.
**Fix**: Either implement the check or delete the test. Don't ship tests that claim to verify properties they don't verify:
```python
# Option 1: Actually verify
import json
result = canonical_json(data)
parsed = json.loads(result)
# Verify no space after colon/comma by checking length
# (canonical form is maximally compact)
re_serialized_compact = json.dumps(parsed, separators=(',', ':'), ensure_ascii=False, sort_keys=True)
assert len(result) <= len(re_serialized_compact), "Canonical JSON not compact"

# Option 2: Delete test (rfc8785 handles this)
```
**Priority**: P1

### Test: test_list_order_matters (line 382)
**Issue**: Test contains multiple early-return escape hatches that drastically reduce effective example count
**Evidence**: Lines 387-394 contain three separate `return` statements that skip test execution. With `max_examples=100`, the effective test count is likely <50 due to identical elements, palindromes, and single-element lists.
**Fix**: Use `assume()` statements instead of early returns to let Hypothesis discard invalid cases and generate more valid examples:
```python
@given(values=st.lists(json_primitives, min_size=2, max_size=10))
@settings(max_examples=100)
def test_list_order_matters(self, values: list[Any]) -> None:
    """Property: List hash depends on element order."""
    # Use assume() to let Hypothesis find valid examples
    assume(len({str(v) for v in values}) > 1)  # Not all identical
    reversed_values = list(reversed(values))
    assume(values != reversed_values)  # Not a palindrome

    hash1 = stable_hash({"list": values})
    hash2 = stable_hash({"list": reversed_values})
    assert hash1 != hash2, "List hash should depend on element order"
```
**Priority**: P2

### Test: test_different_data_different_hash (line 160)
**Issue**: Test validates hash collision resistance but lacks statistical rigor
**Evidence**: Test runs 300 examples expecting zero collisions. SHA-256 has 2^256 space; probability of collision in 300 examples is negligible, making this test essentially a smoke test rather than a property verification. Additionally, this property is guaranteed by SHA-256 mathematics, not our code.
**Fix**: Either delete (redundant with SHA-256 guarantees) or convert to meaningful property test:
```python
# Option 1: Delete (SHA-256 collision resistance is external guarantee)

# Option 2: Test that we're actually USING the hash, not returning constants
@given(data=json_values)
@settings(max_examples=100)
def test_hash_depends_on_input(self, data: Any) -> None:
    """Property: Hash function actually processes input (not constant)."""
    # Modify data and verify hash changes
    if isinstance(data, dict) and data:
        modified = {**data, "__test_mutation__": "changed"}
        assert stable_hash(data) != stable_hash(modified)
    elif isinstance(data, list) and data:
        modified = [*data, "__test_mutation__"]
        assert stable_hash(data) != stable_hash(modified)
```
**Priority**: P3

## Misclassified Tests

### Test: test_hash_version_parameter_works (line 176)
**Issue**: This is a unit test, not a property test
**Evidence**: Test verifies API contract (version parameter accepted, default matches explicit). Property testing adds no value here - the contract is deterministic and doesn't benefit from random examples.
**Fix**: Move to `tests/core/test_canonical.py`:
```python
def test_stable_hash_version_parameter(self) -> None:
    """Version parameter is accepted and default matches v1."""
    data = {"key": "value"}
    hash_default = stable_hash(data)
    hash_explicit = stable_hash(data, version=CANONICAL_VERSION)
    assert hash_default == hash_explicit
```
**Priority**: P2

### Test: test_canonical_json_returns_string (line 90)
**Issue**: Type contract testing - property testing adds minimal value over single unit test
**Evidence**: Testing that return type is `str` for 500 random examples is overkill. Python's type system and one example would suffice.
**Fix**: Move to unit tests or delete (mypy already verifies this)
**Priority**: P3

## Infrastructure Gaps

### Gap: No shrinking validation
**Issue**: Zero tests verify Hypothesis shrinking produces minimal failing examples
**Evidence**: When a property test fails, Hypothesis shrinks to simplest case. If generators don't support shrinking properly, debugging is painful. No test validates this critical feature.
**Fix**: Add shrinking regression tests:
```python
class TestGeneratorShrinking:
    """Verify generators shrink to minimal examples on failure."""

    def test_json_values_shrinks_to_simple_types(self) -> None:
        """When nested structure fails, should shrink to simplest reproduction."""
        from hypothesis import find

        # Find simplest value that would trigger a hypothetical failure
        # (using a fake condition to demonstrate shrinking)
        result = find(
            json_values,
            lambda x: isinstance(x, dict) and len(x) > 0
        )
        # Shrinking should produce minimal dict: single key-value pair
        assert isinstance(result, dict)
        assert len(result) == 1
```
**Priority**: P1

### Gap: No NaN/Infinity rejection property tests
**Issue**: Unit tests verify rejection, but property tests don't generate and verify rejection behavior
**Evidence**: CLAUDE.md states "Test cases must cover: `numpy.int64`, `numpy.float64`, `pandas.Timestamp`, `NaT`, `NaN`, `Infinity`." Unit tests verify this, but property tests generate with `allow_nan=False, allow_infinity=False`, never testing rejection paths.
**Fix**: Add property tests for rejection invariants:
```python
@given(
    data=st.dictionaries(
        dict_keys,
        st.floats(allow_nan=True, allow_infinity=True),
        min_size=1
    )
)
@settings(max_examples=200)
def test_nan_infinity_always_rejected(self, data: dict[str, float]) -> None:
    """Property: Any data containing NaN/Infinity raises ValueError."""
    assume(any(math.isnan(v) or math.isinf(v) for v in data.values()))

    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(data)

    with pytest.raises(ValueError, match="non-finite"):
        stable_hash(data)

@given(arr=st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1))
@settings(max_examples=200)
def test_numpy_array_nan_infinity_rejected(self, arr: list[float]) -> None:
    """Property: NumPy arrays with NaN/Inf always rejected."""
    assume(any(math.isnan(v) or math.isinf(v) for v in arr))
    np_arr = np.array(arr)

    with pytest.raises(ValueError, match="NaN/Infinity found in NumPy array"):
        canonical_json({"array": np_arr})
```
**Priority**: P0 (security-critical for audit integrity)

### Gap: No Decimal non-finite rejection tests
**Issue**: Implementation rejects non-finite Decimals (line 111-112 of canonical.py), but no property test verifies this
**Evidence**: Unit tests may cover this, but property tests should verify the invariant holds across generated Decimal values.
**Fix**: Add property test:
```python
@given(
    value=st.decimals(allow_nan=True, allow_infinity=True)
)
@settings(max_examples=200)
def test_decimal_non_finite_rejected(self, value: Decimal) -> None:
    """Property: Non-finite Decimals always rejected."""
    assume(not value.is_finite())

    with pytest.raises(ValueError, match="non-finite Decimal"):
        canonical_json({"decimal": value})
```
**Priority**: P1

### Gap: No mutation vulnerability testing
**Issue**: Tests verify determinism, but don't verify immutability of input data
**Evidence**: If `canonical_json()` or `stable_hash()` mutates input dicts/lists, audit trail could be corrupted. No test verifies input preservation.
**Fix**: Add mutation detection tests:
```python
@given(data=st.dictionaries(dict_keys, json_primitives, min_size=1))
@settings(max_examples=200)
def test_canonical_json_does_not_mutate_input(self, data: dict[str, Any]) -> None:
    """Property: canonical_json() must not mutate input data."""
    import copy
    original = copy.deepcopy(data)
    _ = canonical_json(data)
    assert data == original, "Input data was mutated"

@given(data=st.lists(json_primitives, min_size=1))
@settings(max_examples=200)
def test_stable_hash_does_not_mutate_lists(self, data: list[Any]) -> None:
    """Property: stable_hash() must not mutate input lists."""
    import copy
    original = copy.deepcopy(data)
    _ = stable_hash(data)
    assert data == original, "Input list was mutated"
```
**Priority**: P1

### Gap: No cross-type equivalence tests
**Issue**: Tests verify numpy ↔ Python equivalence for individual types, but not for nested structures
**Evidence**: `test_numpy_int64_same_as_python_int` (line 205) tests single values, but doesn't verify property holds in nested dicts/lists.
**Fix**: Add compositional property tests:
```python
@given(
    data=st.dictionaries(
        dict_keys,
        st.integers(min_value=_MIN_SAFE_INT, max_value=_MAX_SAFE_INT),
        min_size=1
    )
)
@settings(max_examples=200)
def test_nested_numpy_int_equivalence(self, data: dict[str, int]) -> None:
    """Property: Nested numpy.int64 produces same hash as nested Python int."""
    numpy_data = {k: np.int64(v) for k, v in data.items()}
    hash_numpy = stable_hash(numpy_data)
    hash_python = stable_hash(data)
    assert hash_numpy == hash_python
```
**Priority**: P2

### Gap: No timezone handling property tests
**Issue**: `test_naive_datetime_treated_as_utc` (line 320) is good, but doesn't test other timezones
**Evidence**: Implementation converts all timezones to UTC (line 105 canonical.py), but property tests only verify naive vs UTC, not arbitrary timezone handling.
**Fix**: Add timezone equivalence tests:
```python
# Requires pytz or zoneinfo
@given(
    year=st.integers(min_value=1970, max_value=2100),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),
    hour=st.integers(min_value=0, max_value=23),
    minute=st.integers(min_value=0, max_value=59),
    tz_offset_hours=st.integers(min_value=-12, max_value=14),
)
@settings(max_examples=100)
def test_all_timezones_normalized_to_utc(
    self, year: int, month: int, day: int, hour: int, minute: int, tz_offset_hours: int
) -> None:
    """Property: Same UTC instant from different timezones produces same hash."""
    from datetime import timezone, timedelta

    # Create timezone with offset
    tz = timezone(timedelta(hours=tz_offset_hours))
    dt_tz = datetime(year, month, day, hour, minute, tzinfo=tz)

    # Convert to UTC explicitly
    dt_utc = dt_tz.astimezone(UTC)

    # Both should hash identically (same instant)
    hash_tz = stable_hash({"datetime": dt_tz})
    hash_utc = stable_hash({"datetime": dt_utc})
    assert hash_tz == hash_utc
```
**Priority**: P2

### Gap: No empty container tests
**Issue**: Empty dicts and lists are valid edge cases not explicitly tested
**Evidence**: Generators use `min_size=1` in most places (line 63, 107, etc.), avoiding empty containers.
**Fix**: Add empty container tests or adjust generators:
```python
@given(data=st.dictionaries(dict_keys, json_primitives, min_size=0, max_size=10))
@settings(max_examples=100)
def test_empty_dict_deterministic(self, data: dict[str, Any]) -> None:
    """Property: Empty dicts hash deterministically (edge case)."""
    # Allow Hypothesis to generate empty dicts
    hash1 = stable_hash(data)
    hash2 = stable_hash(data)
    assert hash1 == hash2

def test_empty_containers_explicit(self) -> None:
    """Empty containers must hash consistently."""
    assert stable_hash({}) == stable_hash({})
    assert stable_hash([]) == stable_hash([])
    assert stable_hash({}) != stable_hash([])
```
**Priority**: P2

### Gap: No bytes type property tests
**Issue**: `test_bytes_deterministic` (line 279) verifies determinism, but not base64 encoding correctness
**Evidence**: Implementation encodes bytes as `{"__bytes__": base64}` (line 108 canonical.py). Property test should verify round-trip integrity.
**Fix**: Add validation property test:
```python
@given(data=st.binary(min_size=1, max_size=100))
@settings(max_examples=100)
def test_bytes_round_trip_integrity(self, data: bytes) -> None:
    """Property: Bytes encoding preserves content integrity."""
    import json
    result = canonical_json({"data": data})
    parsed = json.loads(result)

    # Verify structure
    assert "__bytes__" in parsed["data"]

    # Verify round-trip
    decoded = base64.b64decode(parsed["data"]["__bytes__"])
    assert decoded == data
```
**Priority**: P2

### Gap: No max depth testing
**Issue**: `json_values` strategy uses `max_leaves=50` but doesn't verify behavior at max nesting
**Evidence**: Deep nesting can cause stack overflows or performance issues. No test validates behavior at boundary.
**Fix**: Add depth boundary test:
```python
def test_deeply_nested_structure_deterministic(self) -> None:
    """Property: Deeply nested structures hash deterministically."""
    # Create deeply nested structure (50 levels)
    data: dict[str, Any] = {"value": 42}
    for i in range(49):
        data = {"nested": data}

    hash1 = stable_hash(data)
    hash2 = stable_hash(data)
    assert hash1 == hash2

@given(depth=st.integers(min_value=1, max_value=100))
@settings(max_examples=20, deadline=None)  # Deep nesting is slow
def test_arbitrary_depth_deterministic(self, depth: int) -> None:
    """Property: Arbitrary nesting depth produces deterministic hashes."""
    data: dict[str, Any] = {"value": 42}
    for _ in range(depth - 1):
        data = {"nested": data}

    hash1 = stable_hash(data)
    hash2 = stable_hash(data)
    assert hash1 == hash2
```
**Priority**: P3

### Gap: No test for repr_hash() fallback
**Issue**: `canonical.py` defines `repr_hash()` for non-canonical data (line 263), but no property tests verify its contract
**Evidence**: Function is documented as fallback for quarantined data. Should have property tests verifying determinism within same Python version.
**Fix**: Add property tests:
```python
@given(data=json_values)
@settings(max_examples=200)
def test_repr_hash_deterministic_same_session(self, data: Any) -> None:
    """Property: repr_hash() is deterministic within same Python session."""
    from elspeth.core.canonical import repr_hash
    hash1 = repr_hash(data)
    hash2 = repr_hash(data)
    assert hash1 == hash2

def test_repr_hash_accepts_non_canonical_data(self) -> None:
    """repr_hash() should accept data that canonical_json rejects."""
    from elspeth.core.canonical import repr_hash

    # These should not raise
    _ = repr_hash({"nan": float("nan")})
    _ = repr_hash({"inf": float("inf")})
    _ = repr_hash({"custom": object()})  # Non-serializable
```
**Priority**: P2

### Gap: No dict_key ordering robustness test
**Issue**: `test_dict_key_order_independent` (line 359) uses `random.shuffle()` but doesn't verify against insertion-order dicts
**Evidence**: Python 3.7+ guarantees dict insertion order. Test should verify hash independence from dict() vs comprehension vs literal construction methods.
**Fix**: Add construction method independence test:
```python
@given(items=st.lists(st.tuples(dict_keys, json_primitives), min_size=2, max_size=10, unique_by=lambda x: x[0]))
@settings(max_examples=100)
def test_dict_construction_method_independent(self, items: list[tuple[str, Any]]) -> None:
    """Property: Dict hash independent of construction method."""
    # Three different construction methods
    dict_from_constructor = dict(items)
    dict_from_comprehension = {k: v for k, v in items}
    dict_from_reversed = dict(reversed(items))

    hash1 = stable_hash(dict_from_constructor)
    hash2 = stable_hash(dict_from_comprehension)
    hash3 = stable_hash(dict_from_reversed)

    assert hash1 == hash2 == hash3
```
**Priority**: P2

## Positive Observations

- **Strong generator foundation**: JavaScript-safe integer bounds (_MIN_SAFE_INT, _MAX_SAFE_INT) show awareness of RFC 8785 constraints
- **Good documentation**: Docstrings explain *why* properties matter for audit integrity (e.g., lines 81-83, 138-144)
- **Appropriate max_examples**: 100-500 examples balanced for CI performance vs coverage
- **Type equivalence coverage**: Tests for numpy.int64, numpy.float64, numpy.bool_, pd.Timestamp show practical understanding of pipeline data types
- **Structural properties**: `test_dict_key_order_independent` (line 359) correctly validates critical canonicalization requirement
