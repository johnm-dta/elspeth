# Test Quality Review: test_blob_source.py

## Summary

Test suite for AzureBlobSource demonstrates good structural organization with clear test classes and comprehensive coverage of config validation, format support, and authentication methods. However, critical gaps exist in testing external data boundary behavior, NaN/Infinity handling per canonical standards, mutation testing, and CSV parse error edge cases. Several tests violate the "no assertion-free tests" principle by checking only for non-crash behavior without verifying expected outcomes.

## Poorly Constructed Tests

### Test: test_csv_parse_error_quarantines (line 518)
**Issue**: Assertion-free test that only verifies "didn't crash" without checking actual quarantine behavior.
**Evidence**:
```python
# Should NOT raise - should quarantine instead
rows = list(source.load(ctx))

# Pipeline continues (doesn't crash)
# Note: With on_bad_lines="warn", pandas may parse some rows or quarantine the whole file
# Either way, we should not crash
assert isinstance(rows, list)  # Got results, not a crash
```
This assertion is meaningless - `list()` always returns a list. The test doesn't verify that rows were actually quarantined with proper error messages, whether `ctx.record_validation_error()` was called, or what the `SourceRow` states are.

**Fix**: Assert concrete behavior:
```python
rows = list(source.load(ctx))

# Should get quarantined rows, not valid rows
assert len(rows) > 0, "Should quarantine malformed CSV, not return empty list"
assert all(r.is_quarantined for r in rows), "All rows from malformed CSV should be quarantined"
assert any("parse error" in r.quarantine_error.lower() for r in rows), "Should document parse error"
```
**Priority**: P1 - This tests BUG-BLOB-01 fix, must verify actual quarantine behavior

### Test: test_csv_structural_failure_quarantines_blob (line 544)
**Issue**: Assertion-free test - only checks `len(rows) >= 0` which is always true and meaningless.
**Evidence**:
```python
# Should get one quarantined "row" representing the unparseable blob
assert len(rows) >= 0  # Either empty or quarantined row
# No crash = success
```
This is worse than no test - it gives false confidence. The test doesn't verify quarantine behavior, doesn't check `ctx.record_validation_error()` was called, and `len(rows) >= 0` is a tautology.

**Fix**: Assert deterministic behavior:
```python
rows = list(source.load(ctx))

# Binary garbage should quarantine the entire blob as one "row"
assert len(rows) == 1, "Binary data should quarantine blob as single row"
assert rows[0].is_quarantined is True
assert rows[0].quarantine_destination == "quarantine"
assert "parse error" in rows[0].quarantine_error.lower()
assert "__raw_blob_preview__" in rows[0].row, "Should preserve evidence for audit"
```
**Priority**: P0 - Critical BUG-BLOB-01 regression test has no actual assertions

### Test: test_close_clears_client (line 578)
**Issue**: Accesses internal state `_blob_client` directly, violating encapsulation and creating brittle tests.
**Evidence**:
```python
source.close()
assert source._blob_client is None
```
Per CLAUDE.md prohibition on defensive programming: "Access fields directly (obj.field) not defensively (obj.get('field'))" applies to contracts, not internal implementation details. Testing private attributes creates coupling to implementation.

**Fix**: Test observable behavior, not internal state. Either remove the assertion (close is idempotent, that's the contract) or test that subsequent operations after close() raise appropriate errors.
**Priority**: P2 - Brittle test, not critical functionality

### Test: test_close_is_idempotent (line 572)
**Issue**: Assertion-free test - calls close() twice but doesn't verify anything.
**Evidence**:
```python
source.close()
source.close()  # Should not raise
```
While "should not raise" is a valid assertion, the comment doesn't count as an assertion. The test should explicitly verify idempotence.

**Fix**: Add explicit assertions:
```python
source.close()
# First close succeeds
try:
    source.close()
    # Second close also succeeds
except Exception as e:
    pytest.fail(f"close() should be idempotent, raised: {e}")
```
Or better, test that operations after close() have predictable behavior.
**Priority**: P3 - Minor, but sets bad precedent for assertion-free tests

## Missing Critical Test Cases

### Missing: NaN/Infinity rejection in CSV data
**Issue**: Per CLAUDE.md canonical standards, "NaN and Infinity are strictly rejected, not silently converted." No tests verify this defense-in-depth for sources.
**Evidence**: None of the CSV tests include rows with NaN, Inf, or -Inf values.
**Required tests**:
```python
def test_csv_nan_values_quarantined():
    """CSV rows with NaN should quarantine with clear error, not coerce to null."""
    csv_data = b"id,score\n1,NaN\n2,100\n"
    # ...
    rows = list(source.load(ctx))
    assert len(rows) == 2
    assert rows[0].is_quarantined
    assert "NaN" in rows[0].quarantine_error or "invalid" in rows[0].quarantine_error.lower()
    assert not rows[1].is_quarantined

def test_csv_infinity_values_quarantined():
    """CSV rows with Infinity should quarantine, not coerce."""
    csv_data = b"id,score\n1,Infinity\n2,100\n"
    # Should quarantine, not silently convert to null or large number
```
**Priority**: P1 - Critical auditability requirement from CLAUDE.md

### Missing: JSON array containing non-dict values
**Issue**: `_validate_and_yield()` has logic for "May be non-dict for malformed external data (e.g., JSON arrays containing primitives)" but no tests verify this case.
**Evidence**: Line 494 comment in implementation, but TestAzureBlobSourceJSON has no test for `[1, 2, 3]` or `["a", "b"]`.
**Required test**:
```python
def test_json_array_of_primitives_quarantined():
    """JSON array containing primitives should quarantine each element."""
    json_data = b'[1, 2, "string", {"valid": "object"}]'
    # ...
    rows = list(source.load(ctx))
    assert len(rows) == 4
    # First 3 should quarantine (primitives)
    assert all(rows[i].is_quarantined for i in range(3))
    # Last is valid dict
    assert not rows[3].is_quarantined
```
**Priority**: P1 - Documented edge case with no coverage

### Missing: Empty blob files
**Issue**: No tests for empty CSV, empty JSON array `[]`, or empty JSONL files.
**Evidence**: All test data in fixtures has content.
**Required tests**:
```python
def test_csv_empty_file():
    """Empty CSV file should yield zero rows, not crash."""
    csv_data = b""
    # Should yield 0 rows

def test_json_empty_array():
    """Empty JSON array should yield zero rows."""
    json_data = b'[]'
    # Should yield 0 rows

def test_jsonl_empty_file():
    """Empty JSONL file should yield zero rows."""
    jsonl_data = b""
    # Should yield 0 rows
```
**Priority**: P2 - Common edge case in production

### Missing: CSV with only headers, no data
**Issue**: CSV with header row but no data rows - untested.
**Required test**:
```python
def test_csv_headers_only():
    """CSV with headers but no data rows should yield zero rows."""
    csv_data = b"id,name,score\n"  # Header only
    # Should yield 0 rows, not crash or yield malformed row
```
**Priority**: P2

### Missing: Unicode handling beyond latin-1
**Issue**: Only one encoding test (latin-1), but no tests for UTF-8 BOM, emoji, CJK characters, surrogate pairs.
**Required tests**:
```python
def test_csv_utf8_with_bom():
    """CSV with UTF-8 BOM should parse correctly."""
    csv_data = b'\xef\xbb\xbfid,name\n1,test\n'  # UTF-8 BOM

def test_csv_emoji_and_cjk():
    """CSV with emoji and CJK characters should preserve them."""
    csv_data = "id,name\n1,测试\n2,🎉\n".encode('utf-8')
```
**Priority**: P2 - Internationalization requirement

### Missing: Schema coercion verification
**Issue**: Per Three-Tier Trust Model, sources are "the ONLY place coercion is allowed." Tests don't verify that `"42"` → `42` coercion actually happens.
**Evidence**: All CSV tests leave data as strings (`dtype=str` in implementation), but no tests verify that schema with `int` fields actually coerces string values.
**Required test**:
```python
def test_csv_schema_coercion():
    """Source schema should coerce string values to target types."""
    csv_data = b"id,score\n1,100\n2,200\n"  # CSV values are strings
    source = AzureBlobSource(
        make_config(
            schema={
                "mode": "strict",
                "fields": ["id: int", "score: int"],
            }
        )
    )
    rows = list(source.load(ctx))
    # Should coerce CSV strings to ints
    assert rows[0].row["id"] == 1  # int, not "1"
    assert isinstance(rows[0].row["id"], int)
```
**Priority**: P1 - Core auditability requirement, must verify trust boundary behavior

### Missing: JSONL with mixed valid/invalid lines
**Issue**: Line 358 test has valid-invalid-valid pattern, but doesn't verify line number accuracy in quarantine metadata.
**Evidence**: Test checks `"line 2" in results[1].quarantine_error` but doesn't verify `__line_number__` field correctness.
**Fix**: Strengthen test:
```python
assert results[1].row["__line_number__"] == 2, "Should track exact line number for audit"
```
**Priority**: P2 - Auditability detail

### Missing: Concurrent load() calls
**Issue**: No test verifies behavior when `load()` is called multiple times or from multiple contexts.
**Required test**:
```python
def test_load_multiple_times_same_source():
    """load() should be idempotent - multiple calls should work."""
    source = AzureBlobSource(make_config())
    rows1 = list(source.load(ctx))
    rows2 = list(source.load(ctx))
    assert rows1 == rows2  # Same results
```
**Priority**: P3 - Unclear if this is supported, but should be documented

## Misclassified Tests

### Test: test_auth_managed_identity_uses_default_credential (line 765)
**Issue**: Classified as unit test but requires azure.identity package (integration dependency).
**Evidence**: Uses `pytest.importorskip("azure.identity")` in fixture - this makes it an integration test.
**Fix**: Move TestAzureBlobSourceAuthClientCreation class to `tests/integration/test_azure_blob_auth.py` or mark with `@pytest.mark.integration`.
**Priority**: P2 - Organizational, affects test pyramid

### Test: test_auth_service_principal_uses_client_secret_credential (line 793)
**Issue**: Same as above - integration test masquerading as unit test.
**Fix**: Move to integration test suite.
**Priority**: P2

### Test: test_auth_connection_string_uses_from_connection_string (line 827)
**Issue**: Same as above.
**Fix**: Move to integration test suite.
**Priority**: P2

## Infrastructure Gaps

### Gap: No property-based testing for schema validation
**Issue**: Per CLAUDE.md Technology Stack, "Property Testing: Hypothesis - Manual edge-case hunting." The validation tests are all hand-written examples, but Hypothesis would catch edge cases automatically.
**Evidence**: TestAzureBlobSourceValidation has 2 tests - could be 1 property test.
**Recommendation**:
```python
from hypothesis import given, strategies as st

@given(st.dictionaries(st.text(), st.one_of(st.integers(), st.text(), st.none())))
def test_validation_handles_arbitrary_dicts(row_dict):
    """Schema validation should never crash, always quarantine or accept."""
    # Property: validation always returns SourceRow, never raises
    results = list(source._validate_and_yield(row_dict, ctx))
    assert len(results) == 1
    assert isinstance(results[0], SourceRow)
```
**Priority**: P2 - Would catch many edge cases, but hand-written tests are adequate for now

### Gap: Repeated mock setup
**Issue**: Every test manually creates `mock_client` and configures `download_blob().readall()`. This should be a parameterized fixture.
**Evidence**: Lines 212-214, 228-230, 241-243, 253-256, etc. - identical mock setup in every test.
**Fix**: Create fixture:
```python
@pytest.fixture
def mock_blob_data(mock_blob_client):
    """Fixture that returns a function to set blob data."""
    def _set_data(data: bytes):
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = data
        mock_blob_client.return_value = mock_client
    return _set_data

# Then use:
def test_load_csv(mock_blob_data, ctx):
    mock_blob_data(b"id,name\n1,alice\n")
    source = AzureBlobSource(make_config())
    rows = list(source.load(ctx))
```
**Priority**: P3 - DRY principle, but not affecting correctness

### Gap: No fixture for PluginContext with validation tracking
**Issue**: Tests call `ctx.record_validation_error()` but don't verify it was called. The `ctx` fixture is minimal.
**Evidence**: Line 36-38 creates minimal context, but validation tests (line 420, 457) don't verify context was updated.
**Fix**: Create mock context fixture that tracks calls:
```python
@pytest.fixture
def ctx_with_tracking():
    """Context that tracks validation errors."""
    context = MagicMock(spec=PluginContext)
    context.run_id = "test-run"
    context.config = {}
    context.validation_errors = []  # Track calls
    context.record_validation_error = MagicMock(side_effect=lambda **kw: context.validation_errors.append(kw))
    return context
```
**Priority**: P1 - Validation is critical for auditability, must verify context records it

### Gap: No shared test data constants
**Issue**: Test data is inline in each test. Shared constants would make patterns visible.
**Evidence**: `b"id,name,value\n1,alice,100\n"` appears in multiple places with slight variations.
**Fix**: Create module-level constants:
```python
VALID_CSV_TWO_ROWS = b"id,name,value\n1,alice,100\n2,bob,200\n"
INVALID_CSV_BAD_INT = b"id,name,score\n1,alice,95\n2,bob,bad\n3,carol,92\n"
```
**Priority**: P3 - Maintainability, not correctness

## Test Isolation Issues

### Issue: mock_blob_client fixture modifies global state
**Issue**: The `@patch` decorator in the fixture (line 44) patches `AzureBlobSource._get_blob_client` globally for all tests using the fixture. This is correct usage, but could cause confusion if tests are run in unexpected order.
**Evidence**: Line 44 uses `with patch(...)` as context manager, which is scoped correctly.
**Assessment**: Actually fine - using context manager, so properly isolated. Not a real issue.
**Priority**: N/A - False alarm

## Positive Observations

- **Excellent test organization**: Test classes clearly separate concerns (protocol, config validation, CSV, JSON, JSONL, validation, errors, lifecycle, auth).
- **Good use of fixtures**: `make_config()` helper reduces duplication and makes auth option testing clean.
- **Comprehensive auth testing**: All three auth methods tested with mutual exclusivity checks - thorough.
- **BUG-BLOB-01 coverage**: Tests exist for the CSV parse bug (even if assertions are weak).
- **Quarantine testing**: Tests verify both `quarantine` and `discard` modes work correctly.
- **Encoding coverage**: At least one non-UTF-8 test (latin-1) shows awareness of encoding issues.
- **JSONL edge case**: Line 346 tests empty line handling - good attention to detail.

## Risk Assessment

**High Risk**:
- P0 assertion-free tests for BUG-BLOB-01 (lines 518, 544) - regression tests have no teeth
- P1 missing NaN/Infinity rejection tests - violates canonical standards
- P1 missing coercion verification - can't prove trust boundary works correctly
- P1 missing context validation tracking - can't prove audit trail is populated

**Medium Risk**:
- P2 misclassified integration tests - affects CI/CD pipeline structure
- P2 missing empty file tests - production edge case
- P2 missing non-dict JSON handling test - documented but untested

**Low Risk**:
- P3 infrastructure gaps (DRY, fixtures) - maintainability, not correctness
- P3 assertion-free idempotence test - minor issue

## Recommendations

1. **Immediate (P0)**: Fix assertion-free BUG-BLOB-01 tests with concrete quarantine behavior checks.
2. **Pre-release (P1)**: Add NaN/Infinity rejection tests, schema coercion verification, and context validation tracking.
3. **Post-release (P2)**: Reorganize auth tests into integration suite, add empty file tests.
4. **Backlog (P3)**: Refactor fixtures to reduce duplication, consider Hypothesis for property testing.
