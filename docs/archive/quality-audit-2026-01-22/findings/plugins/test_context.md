# Test Quality Review: test_context.py

## Summary

Tests for PluginContext are structurally sound but have significant gaps in verifying contract completeness, immutability guarantees, and concurrent access patterns. Missing validation of Phase 6 features (call recording, audited clients) and weak testing of checkpoint state isolation between nodes.

## Poorly Constructed Tests

### Test: test_get_config_value (line 38)
**Issue**: Incomplete assertion coverage for nested config access
**Evidence**: Test checks `ctx.get("nested.key")` returns `"value"`, but doesn't verify behavior when intermediate keys are missing (e.g., `ctx.get("nested.nonexistent.key")`) or when intermediate values are non-dict types (e.g., `{"nested": "string"}` followed by `ctx.get("nested.key")`).
**Fix**: Add test cases for:
- Missing intermediate keys: `ctx.get("nested.missing.key", default="X")` should return `"X"`
- Non-dict intermediate values: `ctx.get("threshold.nested")` when `threshold` is scalar should return default
- Deep nesting (3+ levels)
**Priority**: P2

### Test: test_checkpoint_typical_batch_workflow (line 105)
**Issue**: Doesn't verify checkpoint isolation between nodes
**Evidence**: Test uses single context with `node_id` not set. According to implementation (lines 138-141 in context.py), checkpoints are keyed by `node_id` to support multiple batch transforms. Test doesn't verify that two contexts with different `node_id` values maintain separate checkpoint state.
**Fix**: Add test creating two contexts with different `node_id` values, update checkpoints independently, verify no cross-contamination.
**Priority**: P1

### Test: test_record_validation_error_generates_row_id_from_hash (line 200)
**Issue**: Weak assertion - only checks length, not hash stability
**Evidence**: `assert len(token.row_id) == 16` doesn't verify that same input produces same hash (determinism) or that different inputs produce different hashes (collision avoidance).
**Fix**: Call `record_validation_error` twice with identical row data, verify `token.row_id` matches. Call with modified row, verify different `row_id`.
**Priority**: P2

### Test: test_update_checkpoint_merges_data (line 81)
**Issue**: Doesn't verify merge semantics for key conflicts
**Evidence**: Test calls `update_checkpoint({"batch_id": "batch-123"})` then `update_checkpoint({"status": "submitted"})` with non-overlapping keys. Doesn't test what happens when same key is updated twice (last-write-wins?), or if nested dicts are deep-merged or shallow-replaced.
**Fix**: Add assertions for:
- Overwrite behavior: `update_checkpoint({"x": 1})`, `update_checkpoint({"x": 2})`, verify `get_checkpoint()["x"] == 2`
- Nested dict behavior: `update_checkpoint({"a": {"b": 1}})`, `update_checkpoint({"a": {"c": 2}})`, verify result
**Priority**: P1

### Test: test_record_validation_error_without_landscape_logs_warning (line 163)
**Issue**: Assertion is too loose - "no landscape" could match unrelated log messages
**Evidence**: `assert "no landscape" in caplog.text.lower()` could match false positives if other code logs similar messages.
**Fix**: Use structured log assertions checking logger name, log level, and message content more precisely. Example: `assert any("no landscape" in rec.message.lower() and rec.levelno == logging.WARNING for rec in caplog.records)`.
**Priority**: P3

## Misclassified Tests

### Test: TestPluginContext (entire class)
**Issue**: These should be contract tests, not unit tests
**Evidence**: `TestPluginContext` verifies the API contract that plugins depend on (method existence, return types, behavior guarantees). This is a cross-cutting concern - if the contract changes, all plugins break. Should live in `tests/contracts/` not `tests/plugins/`.
**Fix**: Move to `tests/contracts/test_plugin_context_contract.py` and structure as contract verification suite with clear sections for required methods, optional integrations, and behavioral guarantees.
**Priority**: P1

### Test: TestCheckpointAPI (entire class)
**Issue**: Mix of unit tests and contract tests
**Evidence**: Tests like `test_checkpoint_methods_exist` (line 53) are contract tests (method presence). Tests like `test_update_checkpoint_stores_data` (line 69) are behavioral tests (implementation verification). These belong in different files with different purposes.
**Fix**: Split into:
- `tests/contracts/test_plugin_context_contract.py`: Method existence, signature verification
- `tests/plugins/test_context.py`: Behavioral verification with full integration testing
**Priority**: P1

## Infrastructure Gaps

### Gap: No fixtures for common PluginContext configurations
**Issue**: Every test manually constructs `PluginContext(run_id="...", config={})`. Code duplication and hard to maintain consistent test state.
**Evidence**: Lines 17, 24, 33, 42, 57, 66, 73, 85, 98, 109, 141, 149, 169, 189, 204, 219, 232, 283, 299, 318, 328, 344, 383, 404 all construct contexts identically.
**Fix**: Add pytest fixtures:
```python
@pytest.fixture
def minimal_ctx() -> PluginContext:
    return PluginContext(run_id="test-run", config={})

@pytest.fixture
def ctx_with_landscape(mock_landscape) -> PluginContext:
    return PluginContext(run_id="test-run", config={}, landscape=mock_landscape)

@pytest.fixture
def ctx_with_node(minimal_ctx) -> PluginContext:
    minimal_ctx.node_id = "test_node"
    return minimal_ctx
```
**Priority**: P2

### Gap: No concurrency tests for call recording
**Issue**: Implementation uses `_call_index_lock` (line 240 in context.py) to prevent race conditions when recording calls. No tests verify thread safety.
**Evidence**: Comment in context.py line 121: "Thread safety for call_index increment (INFRA-01 fix)". Tests don't spawn threads to verify atomicity of `record_call()`.
**Fix**: Add test that spawns 10 threads, each calling `ctx.record_call()` 100 times, verify final `_call_index == 1000` and all recorded calls have unique indices.
**Priority**: P0 (thread safety bugs cause silent data corruption in audit trail)

### Gap: No tests for Phase 6 features (audited clients, call recording)
**Issue**: `PluginContext` has `llm_client`, `http_client`, `state_id`, `_call_index`, and `record_call()` method. Zero tests verify these features.
**Evidence**: No tests in test_context.py call `record_call()` or verify `llm_client`/`http_client` integration.
**Fix**: Add `TestCallRecording` class with tests for:
- `record_call()` without `state_id` raises `RuntimeError`
- `record_call()` with `state_id` increments `_call_index` correctly
- `record_call()` without landscape logs warning and returns `None`
- `record_call()` with landscape delegates to `landscape.record_call()`
**Priority**: P0 (call recording is core audit functionality)

### Gap: No property-based tests for config access edge cases
**Issue**: `ctx.get()` method has complex dotted-path logic (lines 171-188 in context.py). Only tested with 2 levels of nesting. Hypothesis testing would catch edge cases.
**Evidence**: No property tests for:
- Empty string key: `ctx.get("")`
- Leading/trailing dots: `ctx.get(".key.")` or `ctx.get("..key")`
- Unicode keys: `ctx.get("über.groß")`
- Very deep nesting (10+ levels)
**Fix**: Add Hypothesis test generating random dotted paths and config structures, verify no crashes and consistent behavior.
**Priority**: P3 (edge cases unlikely in practice but worth catching)

### Gap: No mutation tests for checkpoint isolation
**Issue**: Tests verify checkpoint CRUD but don't verify that external mutation of returned checkpoint dict doesn't affect internal state.
**Evidence**: `get_checkpoint()` returns `dict[str, Any]` (line 125). If implementation returns reference to internal `_checkpoint` dict instead of copy, caller could mutate it.
**Fix**: Add test:
```python
def test_get_checkpoint_returns_copy_not_reference():
    ctx = PluginContext(run_id="run-1", config={})
    ctx.update_checkpoint({"key": "value"})
    checkpoint = ctx.get_checkpoint()
    checkpoint["key"] = "modified"  # Mutate returned dict
    assert ctx.get_checkpoint()["key"] == "value"  # Should be unchanged
```
**Priority**: P1 (mutation vulnerability could corrupt checkpoint state)

### Gap: No tests for ValidationErrorToken/TransformErrorToken defaults
**Issue**: Dataclass default values are untested. If defaults change, could break plugins expecting specific behavior.
**Evidence**: `ValidationErrorToken.destination` defaults to `"discard"` (line 43 in context.py). Only one test verifies this (line 271). No test verifies `error_id` defaults to `None`.
**Fix**: Add explicit tests for all default values:
```python
def test_validation_error_token_defaults():
    token = ValidationErrorToken(row_id="r1", node_id="n1")
    assert token.error_id is None
    assert token.destination == "discard"
```
**Priority**: P3 (minor but ensures contract stability)

### Gap: Incomplete testing of validation error with non-dict row data
**Issue**: Implementation handles non-dict row data (lines 283-299 in context.py) including fallback to `repr_hash()` for non-canonical data. Only tested implicitly.
**Evidence**: Test `test_record_validation_error_generates_row_id_from_hash` passes dict without `id` field, but doesn't test arrays, primitives, or objects with NaN/Infinity that trigger the repr_hash fallback.
**Fix**: Add tests for:
- Array row: `record_validation_error(row=[1, 2, 3], ...)`
- Primitive row: `record_validation_error(row="malformed", ...)`
- Row with NaN: `record_validation_error(row={"x": float('nan')}, ...)` - verify fallback to repr_hash and warning logged
**Priority**: P1 (Tier-3 data handling is critical for audit integrity)

### Gap: No tests for _batch_checkpoints restoration logic
**Issue**: Implementation has `_batch_checkpoints` dict (line 113) that takes precedence over `_checkpoint` in `get_checkpoint()` (lines 138-141). This is for resume scenarios but is completely untested.
**Evidence**: No test creates context with populated `_batch_checkpoints` and verifies precedence.
**Fix**: Add test:
```python
def test_get_checkpoint_prefers_restored_batch_checkpoint():
    ctx = PluginContext(run_id="run-1", config={}, node_id="node_1")
    # Simulate restored batch checkpoint from previous BatchPendingError
    ctx._batch_checkpoints["node_1"] = {"batch_id": "restored"}
    # Also set local checkpoint
    ctx.update_checkpoint({"batch_id": "local"})
    # Should prefer restored checkpoint
    assert ctx.get_checkpoint()["batch_id"] == "restored"
```
**Priority**: P0 (checkpoint resume is critical for crash recovery)

### Gap: No tests for clear_checkpoint cleaning both checkpoint dicts
**Issue**: Implementation clears both `_checkpoint` and `_batch_checkpoints[node_id]` (lines 166-169), but test only verifies `_checkpoint` is cleared.
**Evidence**: `test_clear_checkpoint_removes_all_data` (line 94) doesn't populate `_batch_checkpoints` before calling `clear_checkpoint()`.
**Fix**: Extend test to populate `_batch_checkpoints`, call `clear_checkpoint()`, verify both are cleared.
**Priority**: P1 (stale checkpoint data could cause resume bugs)

## Positive Observations

- **Clear test organization**: Test classes group related functionality well (CheckpointAPI, ValidationErrorRecording, ValidationErrorDestination, RouteToSink, TransformErrorRecording).
- **Good use of mocks**: Tests use `MagicMock` for landscape integration (lines 224, 425) to isolate context behavior from landscape implementation.
- **Explicit testing of degraded mode**: Tests verify context works without landscape (logs warnings, returns tokens without error_id) - important for Phase 2/3 transition.
- **Destination parameter testing**: Thorough coverage of `destination` field in error tokens (lines 256-309) ensures quarantine routing contract is met.
