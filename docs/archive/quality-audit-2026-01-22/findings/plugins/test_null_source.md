# Test Quality Review: test_null_source.py

## Summary
The test file contains 7 tests covering NullSource, a testing utility source plugin. Multiple critical issues exist: defensive patterns violate codebase standards, protocol/contract tests use runtime checks instead of static typing, test isolation is violated with repeated imports, and several contract-critical scenarios are missing coverage.

## Poorly Constructed Tests

### Test: test_null_source_satisfies_protocol (line 34)
**Issue**: Uses runtime `isinstance()` check for protocol compliance, which is a prohibited defensive pattern
**Evidence**:
```python
def test_null_source_satisfies_protocol(self) -> None:
    """NullSource satisfies SourceProtocol."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    # This should not raise - source satisfies protocol
    assert isinstance(source, SourceProtocol)
```
**Fix**: Delete this test entirely. Protocol compliance is verified by static type checking (mypy), not runtime checks. The CLAUDE.md "No Bug-Hiding Patterns" prohibition explicitly forbids using `isinstance()` to suppress errors. If NullSource doesn't satisfy the protocol, mypy will catch it at type check time. Runtime protocol checks hide interface violations.
**Priority**: P0

### Test: test_null_source_has_output_schema (line 42)
**Issue**: Uses prohibited defensive patterns (`hasattr()`, `issubclass()`) to verify contract compliance
**Evidence**:
```python
def test_null_source_has_output_schema(self) -> None:
    """NullSource has an output_schema attribute."""
    from elspeth.contracts import PluginSchema
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    assert hasattr(source, "output_schema")  # ❌ Prohibited defensive pattern
    # output_schema must be a PluginSchema subclass
    assert issubclass(source.output_schema, PluginSchema)  # ❌ Defensive pattern
```
**Fix**: Delete this test. The CLAUDE.md prohibition states: "Do not use .get(), getattr(), hasattr(), isinstance(), or silent exception handling to suppress errors from nonexistent attributes." If `output_schema` is missing or wrong type, that's a bug in NullSource that should crash immediately. Static type checking verifies this at compile time. If you want to test the schema, test its *behavior*, not its existence.
**Priority**: P0

### Test: test_null_source_has_plugin_version (line 69)
**Issue**: Defensive pattern checking for attribute existence instead of verifying contract behavior
**Evidence**:
```python
def test_null_source_has_plugin_version(self) -> None:
    """NullSource has a plugin_version."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    assert hasattr(source, "plugin_version")  # ❌ Prohibited
    assert isinstance(source.plugin_version, str)  # ❌ Prohibited
    assert source.plugin_version != ""
```
**Fix**: Replace with direct access that verifies the *value*, not the existence:
```python
def test_null_source_plugin_version(self) -> None:
    """NullSource has semantic version."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    assert source.plugin_version == "1.0.0"  # Verify actual value
```
If `plugin_version` doesn't exist or is wrong type, the test crashes - which is correct behavior per CLAUDE.md. The current test uses defensive patterns to hide interface violations.
**Priority**: P0

### Test: test_null_source_has_name (line 27)
**Issue**: No parameterized testing for config edge cases; doesn't verify name is stable across instances
**Evidence**:
```python
def test_null_source_has_name(self) -> None:
    """NullSource has 'null' as its name."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    assert source.name == "null"
```
**Fix**: This test is acceptable but incomplete. Add verification that name is a class attribute (not instance-dependent) and test with different config values:
```python
def test_null_source_name_is_class_attribute(self) -> None:
    """NullSource.name is 'null' regardless of config."""
    from elspeth.plugins.sources.null_source import NullSource

    assert NullSource.name == "null"  # Class-level access
    assert NullSource({}).name == "null"
    assert NullSource({"arbitrary": "config"}).name == "null"
```
**Priority**: P2

## Missing Contract Verification Tests

### Missing: test_null_source_lifecycle_hooks
**Issue**: No tests verify that lifecycle hooks (on_start, on_complete) exist and are callable
**Evidence**: SourceProtocol defines optional hooks; NullSource must provide them (even as no-ops)
**Fix**: Add test:
```python
def test_null_source_lifecycle_hooks_exist(self, ctx: PluginContext) -> None:
    """NullSource has callable lifecycle hooks."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    # Direct access - if hooks don't exist, test crashes (correct per CLAUDE.md)
    source.on_start(ctx)  # Should not raise
    source.on_complete(ctx)  # Should not raise
```
**Priority**: P1

### Missing: test_null_source_node_id_assignment
**Issue**: No test verifies that `node_id` attribute exists and can be set by orchestrator
**Evidence**: SourceProtocol requires `node_id: str | None` attribute
**Fix**: Add test:
```python
def test_null_source_node_id_settable(self) -> None:
    """NullSource.node_id can be set by orchestrator."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    assert source.node_id is None  # Initial state
    source.node_id = "source_001"
    assert source.node_id == "source_001"
```
**Priority**: P2

### Missing: test_null_source_config_validation
**Issue**: No tests verify NullSource behavior with malformed/missing config
**Evidence**: NullSource accepts `dict[str, Any]` config but docstring says "ignored"
**Fix**: Add test to verify config is truly optional:
```python
def test_null_source_ignores_config(self, ctx: PluginContext) -> None:
    """NullSource produces same behavior regardless of config."""
    from elspeth.plugins.sources.null_source import NullSource

    source_empty = NullSource({})
    source_junk = NullSource({"invalid": "garbage", "nested": {"junk": 123}})

    # Both should behave identically
    assert list(source_empty.load(ctx)) == []
    assert list(source_junk.load(ctx)) == []
    assert source_empty.determinism == source_junk.determinism
```
**Priority**: P2

### Missing: test_null_source_load_is_generator
**Issue**: Test verifies `list(source.load(ctx)) == []` but doesn't verify load() returns an iterator
**Evidence**: SourceProtocol.load() must return `Iterator[SourceRow]`, not a list
**Fix**: Add test:
```python
def test_null_source_load_returns_iterator(self, ctx: PluginContext) -> None:
    """load() returns iterator, not list."""
    from collections.abc import Iterator
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    result = source.load(ctx)

    # Direct access - if wrong type, test crashes (correct per CLAUDE.md)
    assert isinstance(result, Iterator)  # This is legitimate - verifying external interface
    assert list(result) == []  # Consuming iterator yields nothing
```
**Priority**: P2

### Missing: test_null_source_multiple_load_calls
**Issue**: No test verifies behavior if load() is called multiple times
**Evidence**: Stateless source should allow repeated calls
**Fix**: Add test:
```python
def test_null_source_load_multiple_calls(self, ctx: PluginContext) -> None:
    """load() can be called multiple times on same instance."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})

    assert list(source.load(ctx)) == []
    assert list(source.load(ctx)) == []  # Second call should work identically
```
**Priority**: P3

### Missing: test_null_source_close_after_load_failure
**Issue**: No test verifies close() can be called after load() raises exception
**Evidence**: SourceProtocol requires close() to clean up resources even on error
**Fix**: While NullSource.load() never raises, this should be documented:
```python
def test_null_source_close_without_load(self) -> None:
    """close() can be called without ever calling load()."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})
    source.close()  # Should not raise even if load() never called
```
**Priority**: P3

## Infrastructure Gaps

### Gap: Repeated imports in every test method
**Issue**: Every test imports NullSource individually, violating DRY and slowing test execution
**Evidence**: All 7 tests contain `from elspeth.plugins.sources.null_source import NullSource`
**Fix**: Move import to module level or create a fixture:
```python
"""Tests for NullSource - a source that yields nothing for resume operations."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol
from elspeth.plugins.sources.null_source import NullSource  # ← Move here


class TestNullSource:
    """Tests for NullSource."""

    @pytest.fixture
    def source(self) -> NullSource:
        """Create NullSource instance for tests."""
        return NullSource({})

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_null_source_yields_nothing(self, source: NullSource, ctx: PluginContext) -> None:
        """NullSource.load() yields no rows."""
        rows = list(source.load(ctx))
        assert rows == []
```
This eliminates import overhead, improves test isolation (each test gets fresh instance), and makes tests more maintainable.
**Priority**: P1

### Gap: No pytest parametrize for determinism/version tests
**Issue**: Tests verify single attribute values without exploring edge cases or variations
**Evidence**: `test_null_source_has_determinism` only checks one enum value
**Fix**: While NullSource only has one valid state, tests should document *why* this is the only valid state:
```python
def test_null_source_determinism(self) -> None:
    """NullSource is DETERMINISTIC - always yields nothing."""
    from elspeth.contracts import Determinism

    # NullSource is deterministic because output never varies
    # (empty iterator has no randomness)
    assert NullSource.determinism == Determinism.DETERMINISTIC

    # Verify this is class-level, not instance-dependent
    source1 = NullSource({})
    source2 = NullSource({"config": "different"})
    assert source1.determinism == source2.determinism == Determinism.DETERMINISTIC
```
**Priority**: P3

### Gap: No comparison with other source plugins
**Issue**: Tests don't verify NullSource behaves differently from real sources (like CSVSource)
**Evidence**: Looking at test_csv_source.py, CSVSource tests verify actual row data. NullSource should explicitly test that it does NOT yield data.
**Fix**: Add explicit comparison test:
```python
def test_null_source_differs_from_real_sources(self, ctx: PluginContext) -> None:
    """NullSource yields nothing unlike real sources."""
    from elspeth.plugins.sources.null_source import NullSource

    source = NullSource({})

    # Unlike CSVSource/JSONSource/BlobSource, NullSource NEVER yields rows
    rows = list(source.load(ctx))
    assert rows == []
    assert len(rows) == 0

    # Verify this is true even if load() called multiple times
    assert list(source.load(ctx)) == []
```
This documents the key behavioral difference from other sources.
**Priority**: P3

## Misclassified Tests

### Test: All tests in this file are correctly classified as unit tests
**Issue**: None - these are appropriate unit tests
**Evidence**: Tests directly instantiate NullSource, provide mocked context, verify isolated behavior without database/filesystem dependencies
**Fix**: None needed
**Priority**: N/A

## Positive Observations

1. **Good test naming**: Test names clearly describe what they verify (`test_null_source_yields_nothing` is unambiguous)
2. **Proper fixture usage**: Uses pytest fixtures for ctx creation (though could be improved with source fixture)
3. **Docstrings present**: Every test has a concise docstring
4. **No sleepy assertions**: Tests have no `time.sleep()` or polling waits
5. **Tests are isolated**: Each test can run independently without setup dependencies
6. **Close idempotency test**: `test_null_source_close_is_idempotent` correctly verifies multiple close() calls are safe

## Confidence Assessment

**Confidence Level**: High (85%)

This review is based on:
- Direct inspection of test code against CLAUDE.md standards
- Comparison with SourceProtocol contract requirements
- Review of comparable test_csv_source.py patterns
- Analysis of NullSource implementation

**Information Gaps**:
1. Unknown if there are integration tests elsewhere that verify NullSource behavior in actual pipeline resume scenarios
2. Unknown if pytest hooks or conftest.py enforce anti-defensive-pattern rules
3. No visibility into whether mypy strict mode is enabled in CI (would catch missing protocol attributes)

## Risk Assessment

**High Risk** (tests hide bugs):
- P0 issues use defensive patterns that could hide interface violations in production
- If NullSource.output_schema implementation changes, tests won't catch breakage
- Runtime `isinstance()` checks bypass type system safety

**Medium Risk** (incomplete coverage):
- Missing lifecycle hook tests mean orchestrator integration could fail
- No verification that load() returns proper iterator type

**Low Risk** (quality/maintenance):
- Repeated imports slow test execution but don't affect correctness
- Missing parametrization makes tests harder to extend

## Caveats

1. **This review applies CLAUDE.md standards strictly**: The prohibition on defensive patterns is unusually strict compared to typical Python codebases. If this standard changes, several findings become invalid.

2. **NullSource is a testing utility, not production code**: Some reviewers might argue that test plugins don't need the same rigor as production sources. However, CLAUDE.md states "Plugin code is reviewed with the same rigor as engine code" - NullSource is still system code.

3. **Static typing vs runtime checks**: This review assumes mypy strict mode is enabled. If not, some of the "delete runtime checks" recommendations may be too aggressive.

4. **Protocol compliance via Protocol vs ABC**: The review assumes runtime_checkable protocols are for static checking only. If the codebase intentionally uses runtime protocol checks for plugin validation, the P0 findings need re-evaluation.
