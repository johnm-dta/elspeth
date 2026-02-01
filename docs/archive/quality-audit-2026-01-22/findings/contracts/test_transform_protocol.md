# Test Quality Review: test_transform_protocol.py

## Summary

Contract test suite has **major completeness gaps** and several **defensive programming anti-patterns**. Tests verify basic protocol attributes but miss critical behavioral contracts (batch-aware processing, error routing, lifecycle edge cases, mutation safety) and use bug-hiding patterns like `hasattr()` checks that violate the "Plugin Ownership: System Code" principle.

## Poorly Constructed Tests

### Test: test_transform_has_* attributes (lines 81-115)
**Issue**: Defensive `hasattr()` checks violate "No Bug-Hiding Patterns" prohibition
**Evidence**:
```python
def test_transform_has_name(self, transform: TransformProtocol) -> None:
    assert hasattr(transform, "name")  # BUG-HIDING PATTERN
    assert isinstance(transform.name, str)
    assert len(transform.name) > 0
```
**Fix**: Direct attribute access - transforms are system code, missing attributes are bugs to crash on:
```python
def test_transform_has_name(self, transform: TransformProtocol) -> None:
    """Contract: Transform MUST have a 'name' attribute."""
    assert isinstance(transform.name, str)  # Let it crash if missing
    assert len(transform.name) > 0
```
**Priority**: P1 - violates codebase philosophy, but tests pass (hides bugs in tests, not in production)

### Test: test_on_start_does_not_raise (line 200)
**Issue**: Defensive `hasattr()` check AND tests wrong contract
**Evidence**:
```python
def test_on_start_does_not_raise(self, transform: TransformProtocol, ctx: PluginContext) -> None:
    if hasattr(transform, "on_start"):  # BUG-HIDING PATTERN
        transform.on_start(ctx)
```
**Fix**: Protocol declares `on_start()` as optional hook (see protocols.py line 199). Either:
1. Remove `hasattr()` and let it crash if missing (protocol contract violation), OR
2. Test that `on_start()` is callable (duck typing), OR
3. Remove test entirely (optional hooks don't need contract tests)
**Priority**: P1 - same anti-pattern as above

### Test: test_on_complete_does_not_raise (line 209)
**Issue**: Same defensive pattern as `test_on_start_does_not_raise`
**Fix**: Same as above
**Priority**: P1

### Test: test_process_handles_extra_fields_gracefully (line 235)
**Issue**: Tests nothing useful - catches all exceptions and declares victory
**Evidence**:
```python
try:
    result = transform.process(input_with_extra, ctx)
    assert isinstance(result, TransformResult)
except Exception:
    pass  # "Some transforms may reject extra fields - that's valid behavior"
```
**Fix**: Either test that transforms **consistently** handle extra fields (success or specific error), or delete the test. Current form asserts "nothing crashes OR it crashes" which is tautological.
**Priority**: P2 - test is useless but not harmful

### Test: test_deterministic_transform_produces_same_output (line 255)
**Issue**: Silent skip if conditions not met - no assertion runs for non-deterministic transforms or error results
**Evidence**:
```python
if transform.determinism == Determinism.DETERMINISTIC:
    result1 = transform.process(valid_input, ctx)
    result2 = transform.process(valid_input, ctx)
    if result1.status == "success" and result2.status == "success" and ...:
        assert result1.row == result2.row  # Only runs if ALL conditions true
```
**Fix**: Split into two tests:
1. `test_deterministic_transform_produces_same_output_on_success` - skip if not DETERMINISTIC
2. `test_deterministic_transform_errors_consistently` - verify error results also deterministic
**Priority**: P2 - limits coverage but doesn't hide bugs

### Test: test_error_result_has_retryable_flag (line 306)
**Issue**: Uses `hasattr()` bug-hiding pattern for contract field
**Evidence**:
```python
assert hasattr(result, "retryable")  # BUG-HIDING PATTERN
```
**Fix**: Direct attribute access - `TransformResult` is a system dataclass:
```python
assert isinstance(result.retryable, bool)  # Let it crash if missing
```
**Priority**: P1 - violates codebase philosophy

## Contract Completeness Gaps

### Missing: Batch-aware transform contract tests
**Issue**: `is_batch_aware` attribute exists (line 107) but no tests verify batch processing behavior
**Evidence**: TransformProtocol.is_batch_aware is a critical flag that changes process() signature from `dict -> TransformResult` to `list[dict] -> TransformResult` (see protocols.py lines 155-156)
**Fix**: Add test class:
```python
class TransformBatchContractTestBase(TransformContractTestBase):
    """Contract tests for batch-aware transforms."""

    @pytest.fixture
    @abstractmethod
    def batch_input(self) -> list[dict[str, Any]]:
        """Provide valid batch input."""
        ...

    def test_batch_aware_flag_is_true(self, transform: TransformProtocol) -> None:
        """Contract: Batch-aware transforms MUST set is_batch_aware=True."""
        assert transform.is_batch_aware is True

    def test_process_accepts_list_input(
        self,
        transform: TransformProtocol,
        batch_input: list[dict[str, Any]],
        ctx: PluginContext
    ) -> None:
        """Contract: Batch-aware transforms MUST accept list[dict] input."""
        result = transform.process(batch_input, ctx)  # type: ignore
        assert isinstance(result, TransformResult)
```
**Priority**: P0 - critical contract gap, untested behavior in production

### Missing: Token creation contract tests
**Issue**: `creates_tokens` attribute exists (line 112) but no tests verify multi-row output behavior
**Evidence**: TransformProtocol.creates_tokens controls whether `success_multi()` creates new tokens (see protocols.py lines 158-162)
**Fix**: Add test:
```python
def test_token_creating_transform_can_emit_multiple_rows(
    self,
    transform: TransformProtocol,
    valid_input: dict[str, Any],
    ctx: PluginContext,
) -> None:
    """Contract: Transforms with creates_tokens=True MAY return success_multi()."""
    if not transform.creates_tokens:
        pytest.skip("Transform does not create tokens")

    result = transform.process(valid_input, ctx)
    if result.status == "success" and result.is_multi_row:
        assert isinstance(result.rows, list)
        assert len(result.rows) > 0
```
**Priority**: P1 - important contract, but less critical than batch-aware

### Missing: Error routing contract tests
**Issue**: `_on_error` attribute exists (protocols.py line 167) but no tests verify error routing configuration
**Evidence**: WP-11.99b requires transforms that return errors to specify `_on_error` destination
**Fix**: Add test:
```python
def test_error_returning_transform_has_error_routing(
    self,
    transform: TransformProtocol,
    error_input: dict[str, Any],
    ctx: PluginContext,
) -> None:
    """Contract: Transforms that return errors MUST configure _on_error."""
    result = transform.process(error_input, ctx)
    if result.status == "error":
        # Transform returned error - must have configured error routing
        assert transform._on_error is not None, (
            "Transform returns error results but _on_error is None. "
            "Set _on_error in config or ensure all errors are bugs (crash instead)."
        )
```
**Priority**: P1 - architectural requirement from WP-11.99b

### Missing: Mutation safety tests
**Issue**: Transforms receive pipeline data (Tier 2 trust) but no tests verify they don't mutate input
**Evidence**: PassThrough explicitly does `copy.deepcopy(row)` (passthrough.py line 87), suggesting mutation is a concern
**Fix**: Add test:
```python
def test_process_does_not_mutate_input(
    self,
    transform: TransformProtocol,
    valid_input: dict[str, Any],
    ctx: PluginContext,
) -> None:
    """Contract: Transforms MUST NOT mutate input row dict."""
    original = copy.deepcopy(valid_input)
    transform.process(valid_input, ctx)
    assert valid_input == original, "Transform mutated input row"
```
**Priority**: P1 - data integrity contract

### Missing: Close idempotency REQUIRES processing first
**Issue**: Test assumes `close()` can be called without `process()`, but this is untested assumption
**Evidence**: Line 191 shows `transform.process(valid_input, ctx)` before close, but what if close() is called without any processing?
**Fix**: Add separate test:
```python
def test_close_without_processing_is_safe(
    self,
    transform: TransformProtocol,
) -> None:
    """Contract: close() MUST be safe even if process() never called."""
    transform.close()  # Should not crash
```
**Priority**: P2 - edge case but important for error handling paths

### Missing: Context object must not be stored
**Issue**: No test verifies transforms don't store PluginContext across calls
**Evidence**: PluginContext has per-call lifecycle (protocols.py line 24), storing it risks stale data
**Fix**: Add test (property-based):
```python
def test_transform_does_not_store_context(
    self,
    transform: TransformProtocol,
    valid_input: dict[str, Any],
) -> None:
    """Contract: Transforms MUST NOT store PluginContext between calls."""
    ctx1 = PluginContext(run_id="run-1", config={}, node_id="node", plugin_name="test")
    ctx2 = PluginContext(run_id="run-2", config={}, node_id="node", plugin_name="test")

    transform.process(valid_input, ctx1)
    result2 = transform.process(valid_input, ctx2)

    # If transform stored ctx1, this might use wrong run_id
    # (Hard to verify without inspecting internals - consider removing if too invasive)
```
**Priority**: P3 - hard to test without inspection, consider skipping

### Missing: Schema validation on success
**Issue**: No test verifies that success result rows match `output_schema`
**Evidence**: TransformProtocol has `output_schema` (line 93) but tests never validate output against it
**Fix**: Add test:
```python
def test_success_output_matches_output_schema(
    self,
    transform: TransformProtocol,
    valid_input: dict[str, Any],
    ctx: PluginContext,
) -> None:
    """Contract: Success result MUST match output_schema."""
    result = transform.process(valid_input, ctx)
    if result.status == "success" and result.row is not None:
        # This will raise ValidationError if output doesn't match schema
        validated = transform.output_schema.model_validate(result.row)
        assert validated is not None
```
**Priority**: P1 - core schema contract, critical for type safety

### Missing: Audit field initialization
**Issue**: No test verifies that audit fields (input_hash, output_hash, duration_ms) start as None
**Evidence**: TransformResult has audit fields (results.py lines 84-86) set by executor, plugins must not touch
**Fix**: Add test:
```python
def test_transform_does_not_set_audit_fields(
    self,
    transform: TransformProtocol,
    valid_input: dict[str, Any],
    ctx: PluginContext,
) -> None:
    """Contract: Plugins MUST NOT set audit fields (input_hash, output_hash, duration_ms)."""
    result = transform.process(valid_input, ctx)
    assert result.input_hash is None, "Plugin must not set input_hash"
    assert result.output_hash is None, "Plugin must not set output_hash"
    assert result.duration_ms is None, "Plugin must not set duration_ms"
```
**Priority**: P1 - audit integrity contract

## Misclassified Tests

No misclassification issues - tests are correctly categorized as contract tests.

## Infrastructure Gaps

### Gap: No concrete test fixtures for common patterns
**Issue**: Every subclass must implement `transform`, `valid_input`, `ctx` fixtures from scratch
**Evidence**: Lines 55-75 show abstract fixtures but no reusable concrete ones
**Fix**: Provide fixture helpers:
```python
# In conftest.py or test utilities
@pytest.fixture
def minimal_plugin_context() -> PluginContext:
    """Minimal PluginContext for contract tests."""
    return PluginContext(
        run_id="test-run-001",
        config={},
        node_id="test-transform",
        plugin_name="test",
    )
```
**Priority**: P3 - reduces boilerplate but doesn't affect correctness

### Gap: No parametrized attribute tests
**Issue**: Lines 81-115 repeat identical pattern 7 times (hasattr + isinstance + assertion)
**Evidence**: Could be single parametrized test
**Fix**:
```python
@pytest.mark.parametrize("attr_name,expected_type", [
    ("name", str),
    ("input_schema", type),
    ("output_schema", type),
    ("determinism", Determinism),
    ("plugin_version", str),
    ("is_batch_aware", bool),
    ("creates_tokens", bool),
])
def test_transform_has_required_attribute(
    self,
    transform: TransformProtocol,
    attr_name: str,
    expected_type: type,
) -> None:
    """Contract: Transform MUST have required protocol attributes."""
    value = getattr(transform, attr_name)  # Let it crash if missing
    assert isinstance(value, expected_type)
```
**Priority**: P3 - reduces repetition but same functionality

### Gap: No test execution order guarantees
**Issue**: Tests call `process()` before `close()` but pytest doesn't guarantee order
**Evidence**: Line 191 assumes `process()` runs before `close()` in same test
**Fix**: Either:
1. Make each test independent (call process in every test that needs it), OR
2. Use pytest-order to enforce sequence
Current approach is actually CORRECT (each test is independent), but documentation could clarify.
**Priority**: P4 - tests are actually correct, just looks confusing

## Positive Observations

- **Abstract base approach is excellent** - forces implementers to provide fixtures, ensures consistency
- **Separate error contract base class** - `TransformErrorContractTestBase` cleanly separates error path testing
- **Property-based testing foundation** - `TransformContractPropertyTestBase` shows commitment to thorough testing
- **Clear docstrings** - Every test explains what contract it verifies
- **Factory method validation** - Tests verify `TransformResult.success()` and `TransformResult.error()` usage

## Summary

**Critical gaps** (P0): 1 (batch-aware processing)
**High-priority gaps** (P1): 8 (defensive patterns, schema validation, mutation safety, error routing, audit fields)
**Medium-priority gaps** (P2): 2 (determinism edge cases, close without processing)
**Low-priority improvements** (P3-P4): 4 (fixtures, parametrization)

**Bottom line**: Contract test suite covers basic protocol shape but misses **behavioral contracts** that distinguish transforms from plain functions. Biggest risks are untested batch-aware behavior and use of bug-hiding patterns that violate project philosophy.
