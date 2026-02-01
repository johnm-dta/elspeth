# Test Quality Review: test_manager_validation.py

## Summary

This test file has been gutted to the point of near-uselessness. It contains only 2 tests for PluginSpec validation when the actual `manager.py` contains extensive validation logic for plugin registration, duplicate detection, schema hashing, config validation, and plugin creation. The test file is incomplete and fails to verify critical plugin manager behaviors.

## Poorly Constructed Tests

### Test: test_schemas_default_to_none (line 30)
**Issue**: Tests obvious implementation detail rather than behavior
**Evidence**: The test checks that `input_schema_hash` and `output_schema_hash` are None when schemas are not provided. This is a trivial default behavior that provides no value.
**Fix**: Replace with meaningful schema validation tests:
- Test that schemas with different field sets produce different hashes
- Test that schema hash is deterministic (same schema → same hash)
- Test that schema hash changes when field types change
- Test that invalid schema types (non-Pydantic) are rejected with TypeError per line 51 of manager.py
**Priority**: P1 - Tests obvious trivia instead of real validation

### Test: test_valid_plugin_succeeds (line 17)
**Issue**: Only tests the happy path with minimal assertions
**Evidence**: Creates a minimal plugin and checks 3 attributes. Does not verify schema hashing, node_type assignment, or edge cases.
**Fix**: Add comprehensive validations:
- Verify that `node_type` is correctly assigned from parameter
- Test with actual schemas (not just None) and verify schema hashing
- Test with different `Determinism` enum values
- Test with edge case version strings (empty, very long, special chars)
**Priority**: P2 - Incomplete coverage of success path

## Missing Critical Tests

### Category: Schema Hash Validation
**Issue**: No tests for `_schema_hash()` function behavior
**Evidence**: `manager.py` lines 30-58 contain critical schema hashing logic that is completely untested
**Missing Tests**:
1. Schema with Pydantic BaseModel produces deterministic hash
2. Same schema class produces identical hash when called twice
3. Different schemas (different fields) produce different hashes
4. Non-Pydantic schema raises TypeError with specific message
5. Schema with changed field type produces different hash
6. Schema with reordered fields produces same hash (order independence)
7. Schema with nested models hashes correctly
**Priority**: P0 - Core validation logic is untested

### Category: PluginManager Registration
**Issue**: No tests for the actual PluginManager class
**Evidence**: The file is named `test_manager_validation.py` but contains zero tests for PluginManager
**Missing Tests**:
1. `register()` method adds plugin and refreshes caches
2. Duplicate plugin names raise ValueError with descriptive message
3. `_refresh_caches()` populates sources/transforms/gates/sinks correctly
4. Multiple plugins of different types can be registered
5. Hookspecs are registered correctly in `__init__`
**Priority**: P0 - Core manager functionality untested

### Category: Plugin Lookup
**Issue**: No tests for get_*_by_name() methods
**Evidence**: Lines 232-302 in manager.py contain lookup methods with error handling
**Missing Tests**:
1. `get_source_by_name()` returns correct plugin class
2. `get_source_by_name()` with unknown name raises ValueError with available plugins list
3. `get_transform_by_name()` lookup succeeds and fails appropriately
4. `get_gate_by_name()` lookup behavior
5. `get_sink_by_name()` lookup behavior
6. Error messages include sorted list of available plugins
**Priority**: P0 - Plugin lookup is core manager functionality

### Category: Plugin Creation with Validation
**Issue**: No tests for create_source/transform/gate/sink methods
**Evidence**: Lines 306-412 contain plugin instantiation with config validation
**Missing Tests**:
1. `create_source()` validates config and instantiates plugin
2. `create_source()` with invalid config raises ValueError with field-specific errors
3. `create_source()` with unknown plugin type fails appropriately
4. Same tests for create_transform/create_gate/create_sink
5. Validation errors are formatted with field names (line 323)
**Priority**: P0 - Config validation is security-critical

### Category: Duplicate Detection
**Issue**: No tests for duplicate plugin name detection
**Evidence**: Lines 181-204 in manager.py check for duplicate names
**Missing Tests**:
1. Registering two sources with same name raises ValueError
2. Registering two transforms with same name raises ValueError
3. Error message includes name and existing plugin class name
4. Same name allowed across different plugin types (source and sink both named "csv")
**Priority**: P0 - Duplicate detection prevents config errors

## Misclassified Tests

This file has no misclassification issues because it barely tests anything. However, when proper tests are added:

**Recommendation**: Split into multiple files:
- `test_plugin_spec.py` - PluginSpec.from_plugin() and _schema_hash()
- `test_plugin_manager_registration.py` - register() and duplicate detection
- `test_plugin_manager_lookup.py` - get_*_by_name() methods
- `test_plugin_manager_creation.py` - create_*() with config validation

## Infrastructure Gaps

### Gap: No Schema Test Fixtures
**Issue**: No reusable test schemas for schema hashing tests
**Evidence**: Tests create inline classes, would duplicate across proper tests
**Fix**: Create fixtures:
```python
@pytest.fixture
def sample_schema():
    class SampleSchema(PluginSchema):
        field_a: str
        field_b: int
    return SampleSchema

@pytest.fixture
def different_schema():
    class DifferentSchema(PluginSchema):
        field_x: float
    return DifferentSchema
```
**Priority**: P1

### Gap: No Plugin Manager Fixture
**Issue**: Each test would need to instantiate PluginManager
**Evidence**: No manager fixture exists
**Fix**: Create fixture with clean state:
```python
@pytest.fixture
def plugin_manager():
    """Clean PluginManager instance."""
    return PluginManager()
```
**Priority**: P1

### Gap: No Mock Plugin Classes
**Issue**: Tests need minimal plugin implementations
**Evidence**: test_valid_plugin_succeeds creates inline class
**Fix**: Create reusable test plugin fixtures for each protocol type
**Priority**: P2

### Gap: No Integration with Actual Plugins
**Issue**: Tests use synthetic plugin classes, not real ones
**Evidence**: All tests use inline class definitions
**Fix**: Add integration tests that verify real built-in plugins:
- Verify NullSource is discovered and registered
- Verify PassthroughTransform is discovered and registered
- Verify schema hashes match for real schemas
**Rationale**: Per CLAUDE.md "Plugins are system-owned code" - we should test that real plugins integrate correctly
**Priority**: P1

## Alignment with CLAUDE.md Standards

### VIOLATION: Defensive Programming Prohibition
**Location**: N/A (no tests violate this)
**Analysis**: File is too sparse to violate standards

### VIOLATION: Plugin Ownership Model
**Issue**: Tests don't verify crash-on-bug behavior
**Evidence**: No tests verify that invalid plugin attributes cause crashes vs silent failures
**Required Tests**:
1. Non-Pydantic schema crashes with TypeError (not silently ignored)
2. Missing required protocol attributes crash at type-check time (verified via mypy, not runtime)
3. Invalid determinism enum value crashes
**Rationale**: Per CLAUDE.md line 168: "Plugin missing expected attribute | **CRASH** - interface violation | Use getattr(x, 'attr', default)"
**Priority**: P1

### ALIGNMENT: Trust Model
**Issue**: No tests verify schema validation treats plugin schemas as "Their Data" (Tier 3)
**Evidence**: PluginSchema uses `extra="ignore"` and `strict=False` per data.py lines 44-46
**Missing Tests**:
- Schema validation allows extra fields in row
- Schema validation coerces types (int → float)
- Schema validation doesn't crash on user data
**Priority**: P2

## Test Pyramid Analysis

**Current Distribution**: 2 unit tests, 0 integration, 0 contract tests

**Problems**:
- Missing contract tests for PluginProtocol adherence
- Missing integration tests with real plugins
- Insufficient unit test coverage (2 tests for ~300 LOC module)

**Target Distribution**:
- Unit: ~20 tests (schema hashing, registration, lookup, creation)
- Integration: ~5 tests (real plugin registration, discovery integration)
- Contract: ~5 tests (verify plugins satisfy protocols)

## Positive Observations

1. **Comment explains mypy enforcement** (lines 4-7): Correctly explains that runtime validation for required attributes is unnecessary due to type checking
2. **No sleepy assertions**: No time.sleep() anti-patterns
3. **No shared state**: Tests are independent (though trivial)

## Overall Assessment

**Test Coverage**: ~5% (2 trivial tests for 300+ line module)

**Critical Gaps**:
- Schema hashing completely untested
- PluginManager core methods completely untested
- Config validation completely untested
- Duplicate detection completely untested
- Plugin lookup completely untested

**Recommended Action**: This file should be considered a stub. Either:
1. Rename to `test_plugin_spec_basic.py` and acknowledge it only tests PluginSpec
2. Expand to comprehensive manager validation suite (40+ tests)
3. Delete and start fresh with proper test design

The current state is misleading - the filename suggests comprehensive manager validation but delivers nearly nothing.
