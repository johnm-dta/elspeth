# Test Quality Review: test_orchestrator_validation.py

## Summary
Test file validates that `_validate_transform_error_sinks()` catches configuration errors at pipeline startup. Tests are structurally sound but suffer from severe code duplication (5 identical test plugin classes repeated across 6 tests, ~270 lines of duplicated boilerplate). Infrastructure gaps prevent efficient testing. Coverage is good for the narrow scope.

## Poorly Constructed Tests

### Test: All tests - Massive Code Duplication (lines 100-577)
**Issue**: Every test defines identical `CollectSink`, `ListSource`/`TrackingSource`, and `InputSchema` classes. Approximately 270 lines of duplicated boilerplate across 6 tests.

**Evidence**:
- `InputSchema` defined identically 6 times (lines 113, 194, 277, 348, 418, 498)
- `CollectSink` (25 lines) defined identically 6 times (lines 144-161, 222-239, 304-321, 375-392, 446-463, 535-555)
- `ListSource`/`TrackingSource` (10-15 lines) defined 5 times (lines 197-206, 279-288, 350-359, 421-430, 506-518)

**Fix**: Create shared fixtures in `conftest.py`:
```python
# tests/conftest.py
@pytest.fixture
def minimal_input_schema():
    """Minimal schema for validation tests."""
    class InputSchema(PluginSchema):
        value: int
    return InputSchema

@pytest.fixture
def collect_sink():
    """Reusable in-memory sink for validation tests."""
    class CollectSink(_TestSinkBase):
        name = "collect"
        def __init__(self):
            self.results = []
        def write(self, rows, ctx):
            self.results.extend(rows)
            return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")
    return CollectSink()

@pytest.fixture
def list_source(minimal_input_schema):
    """Reusable list-based source."""
    class ListSource(_TestSourceBase):
        name = "list_source"
        output_schema = minimal_input_schema
        def __init__(self, data):
            self._data = data
        def load(self, ctx):
            for row in self._data:
                yield SourceRow.valid(row)
    return ListSource
```

Then reduce each test to 15-20 lines of unique logic (transform definition + assertions).

**Priority**: P1 (major maintenance burden, but tests work)

---

### Test: All tests - Duplicated `_build_test_graph` Helper (lines 31-89)
**Issue**: Test file defines `_build_test_graph()` locally, but this is a general-purpose DAG construction utility that other tests likely need.

**Evidence**: 59-line helper function for creating simple linear graphs from `PipelineConfig`.

**Fix**: Move to `tests/conftest.py` or `tests/engine/conftest.py` if other orchestrator tests need it. This helper is reusable across any test that needs a simple graph for orchestrator validation.

**Priority**: P2 (duplication across test files possible)

---

### Test: test_validation_occurs_before_row_processing (lines 485-577)
**Issue**: Uses mutable shared state (`call_tracking` dict) to track method calls instead of instance attributes. This is more fragile and harder to debug than instance-based tracking.

**Evidence**:
```python
call_tracking: dict[str, bool] = {
    "source_load_called": False,
    "transform_process_called": False,
    "sink_write_called": False,
}

class TrackingSource(_TestSourceBase):
    def load(self, ctx: Any) -> Iterator[SourceRow]:
        call_tracking["source_load_called"] = True  # Mutates global dict
```

**Fix**: Use instance attributes on tracking classes:
```python
class TrackingSource(_TestSourceBase):
    def __init__(self, data):
        self._data = data
        self.load_called = False

    def load(self, ctx):
        self.load_called = True
        for row in self._data:
            yield SourceRow.valid(row)
```

Then assert on `source.load_called`, `transform.process_called`, etc. (pattern already used in test at lines 100-181).

**Priority**: P2 (inconsistent with other tests, harder to debug)

---

## Misclassified Tests

### None Identified
Tests are correctly classified as unit tests for validation logic. They properly isolate the validation method by testing it through `orchestrator.run()` without executing actual row processing.

---

## Infrastructure Gaps

### Gap: No Shared Test Plugin Repository (lines 100-577)
**Issue**: Tests define plugins inline, leading to 270+ lines of boilerplate duplication. No centralized repository for common test plugins.

**Evidence**: Every test redeclares `CollectSink`, `ListSource`, `InputSchema` identically.

**Fix**: Create `tests/engine/validation_fixtures.py` or add to `tests/conftest.py`:
```python
# Minimal schema
class MinimalSchema(PluginSchema):
    value: int

# Reusable sink
class CollectSink(_TestSinkBase):
    """In-memory sink for validation tests."""
    name = "collect"
    def __init__(self):
        self.results = []
    # ... methods ...

# Reusable source factory
def make_list_source(data: list[dict[str, Any]], schema_type: type[PluginSchema] = MinimalSchema):
    """Factory for creating list-based sources with custom schemas."""
    class ListSource(_TestSourceBase):
        name = "list_source"
        output_schema = schema_type
        def __init__(self):
            self._data = data
        def load(self, ctx):
            for row in self._data:
                yield SourceRow.valid(row)
    return ListSource()
```

**Priority**: P1 (critical for maintainability)

---

### Gap: No Parametrized Test for Special Values (lines 268-408)
**Issue**: Three separate tests verify special `on_error` values (`"discard"`, `None`, valid sink name) when this should be a single parametrized test.

**Evidence**:
- `test_on_error_discard_passes_validation` (lines 268-338)
- `test_on_error_none_passes_validation` (lines 339-409)
- `test_valid_on_error_sink_passes_validation` (lines 410-484)

All have identical structure (setup, run, assert `result.status == "completed"`).

**Fix**: Replace with parametrized test:
```python
@pytest.mark.parametrize("on_error_value,description", [
    ("discard", "discard special value"),
    (None, "None (no error routing)"),
    ("error_sink", "valid sink name"),
])
def test_on_error_validation_passes(on_error_value, description):
    """on_error validation passes for: {description}."""
    class TestTransform(BaseTransform):
        _on_error = on_error_value
        # ... rest of implementation ...

    sinks = {"default": as_sink(sink)}
    if on_error_value == "error_sink":
        sinks["error_sink"] = as_sink(CollectSink())

    config = PipelineConfig(source=as_source(source), transforms=[transform], sinks=sinks)
    result = orchestrator.run(config, graph=_build_test_graph(config))
    assert result.status == "completed"
```

This reduces 140 lines to ~30 lines with better coverage documentation.

**Priority**: P2 (improves clarity and reduces maintenance)

---

### Gap: Missing DAG Integration Test
**Issue**: Tests use `_build_test_graph()` helper which manually constructs graphs, bypassing actual DAG compilation. No test verifies that real DAG compilation properly exposes error sink validation failures.

**Evidence**: All tests use `graph=_build_test_graph(config)` instead of letting orchestrator compile the graph from config.

**Fix**: Add integration test:
```python
def test_error_sink_validation_with_real_dag_compilation():
    """Validation fails even when using real DAG compilation (not manual graph)."""
    # Same setup as test_invalid_on_error_sink_fails_at_startup
    # but DO NOT pass graph= parameter - let orchestrator compile it

    orchestrator = Orchestrator(db)
    with pytest.raises(RouteValidationError):
        orchestrator.run(config)  # No graph= argument
```

This verifies the validation runs in the real execution path, not just with test harness graphs.

**Priority**: P2 (tests cover intended behavior, but integration gap exists)

---

### Gap: No Boundary Testing for Error Messages
**Issue**: `test_error_message_includes_transform_name_and_sinks` (lines 182-267) verifies message *contains* key terms but doesn't verify message *format* or *completeness*.

**Evidence**:
```python
assert "my_bad_transform" in error_msg
assert "phantom_sink" in error_msg
assert "default" in error_msg
assert "error_archive" in error_msg
```

No verification that message structure matches expected format from implementation:
```
Transform '{transform.name}' has on_error='{on_error}' but no sink named '{on_error}' exists.
Available sinks: {sorted(available_sinks)}. Use 'discard' to drop error rows without routing.
```

**Fix**: Test exact format:
```python
expected_msg = (
    "Transform 'my_bad_transform' has on_error='phantom_sink' "
    "but no sink named 'phantom_sink' exists. "
    "Available sinks: ['default', 'error_archive']. "
    "Use 'discard' to drop error rows without routing."
)
assert str(exc_info.value) == expected_msg
```

Or use regex if ordering of available sinks varies:
```python
import re
pattern = re.compile(
    r"Transform 'my_bad_transform' has on_error='phantom_sink' "
    r"but no sink named 'phantom_sink' exists\. "
    r"Available sinks: \['default', 'error_archive'\]\. "
    r"Use 'discard' to drop error rows without routing\."
)
assert pattern.match(str(exc_info.value))
```

**Priority**: P3 (current test catches regressions in message content)

---

### Gap: No Test for Multiple Invalid Transforms
**Issue**: Tests only verify single transforms with invalid `on_error`. No test verifies behavior when *multiple* transforms have invalid error sinks (does validation fail on first error? report all errors?).

**Evidence**: All tests use `transforms=[single_transform]`.

**Fix**: Add test:
```python
def test_validation_reports_first_invalid_error_sink():
    """Validation fails on first invalid error sink (fail-fast behavior)."""
    class BadTransform1(BaseTransform):
        name = "bad_transform_1"
        _on_error = "nonexistent_1"
        # ...

    class BadTransform2(BaseTransform):
        name = "bad_transform_2"
        _on_error = "nonexistent_2"
        # ...

    config = PipelineConfig(
        source=as_source(source),
        transforms=[BadTransform1(), BadTransform2()],
        sinks={"default": as_sink(sink)},
    )

    with pytest.raises(RouteValidationError) as exc_info:
        orchestrator.run(config, graph=_build_test_graph(config))

    # Verify it reports the FIRST error (bad_transform_1)
    assert "bad_transform_1" in str(exc_info.value)
    # Note: If validation is fail-fast, bad_transform_2 won't be in the message
```

This documents the validation strategy (fail-fast vs. collect-all-errors).

**Priority**: P3 (edge case, current implementation is fail-fast)

---

## Positive Observations

1. **Excellent Test Naming**: Test names are precise and document the expected behavior clearly (`test_invalid_on_error_sink_fails_at_startup`, `test_validation_occurs_before_row_processing`).

2. **Correct Validation Timing**: Tests properly verify validation happens BEFORE row processing (lines 179-180, 573-576), which is critical for the "fail-fast at startup" design.

3. **Complete Coverage of Special Cases**: Tests cover all special `on_error` values (`None`, `"discard"`, valid sink) plus error cases.

4. **Good Use of Tracking Pattern**: `test_invalid_on_error_sink_fails_at_startup` uses instance-based tracking (`source.load_called`) which is clean and debuggable.

5. **Proper Exception Type Checking**: Tests use `pytest.raises(RouteValidationError)` not generic `Exception`, verifying the exact error type.

6. **Documentation**: Docstrings clearly explain what each test verifies, matching test names.

---

## Recommendations Summary

| Priority | Category | Action | Effort | Impact |
|----------|----------|--------|--------|--------|
| P1 | Infrastructure | Create shared test plugin fixtures (CollectSink, ListSource, MinimalSchema) | 2h | Eliminate 270 lines of duplication |
| P1 | Infrastructure | Move `_build_test_graph` to conftest.py | 30min | Enable reuse across test files |
| P2 | Infrastructure | Convert 3 special-value tests to single parametrized test | 1h | Reduce 140 lines to 30 |
| P2 | Poor Construction | Replace mutable `call_tracking` dict with instance attributes | 30min | Improve debuggability |
| P2 | Infrastructure | Add DAG integration test (no manual graph) | 1h | Verify real execution path |
| P3 | Infrastructure | Add exact message format verification | 1h | Document message contract |
| P3 | Infrastructure | Add multi-transform invalid error sink test | 1h | Document fail-fast behavior |

**Total estimated effort**: ~8 hours to address all findings.

**Immediate action**: Create shared fixtures (P1) before expanding test coverage further. Current duplication makes adding tests prohibitively expensive.
