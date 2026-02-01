# Test Quality Review: test_orchestrator_cleanup.py

## Summary

Test suite validates transform cleanup lifecycle but has critical gaps: missing source/sink cleanup tests, no exception propagation verification, and incomplete cleanup ordering validation. Tests focus narrowly on transform.close() while ignoring broader cleanup contract violations in the orchestrator implementation.

## SME Agent Protocol: Confidence Assessment

**Confidence Level:** HIGH (85%)

**Reasoning:**
- Test file fully read and understood
- Orchestrator implementation examined (lines 202-222, 580-588, 1129-1146)
- CLAUDE.md standards reviewed (Plugin Ownership, No Bug-Hiding Patterns)
- BaseTransform/BaseGate close() contracts confirmed

**Caveats:**
- Did not execute tests to verify flakiness behavior
- Did not review complete test coverage metrics for orchestrator.py
- Cleanup sequencing guarantees not fully explored across all failure modes

## SME Agent Protocol: Risk Assessment

**High-Risk Findings:**

1. **Missing source/sink cleanup tests (P0)** - Production resource leaks undetected
2. **No cleanup ordering verification (P1)** - Violates lifecycle contracts silently
3. **Missing exception suppression tests (P1)** - Best-effort cleanup not validated

**Medium-Risk Findings:**

4. **Graph mutation vulnerability (P2)** - Tests directly mutate internal graph state
5. **No multi-sink cleanup tests (P2)** - Production pattern untested

## SME Agent Protocol: Information Gaps

**Missing Information:**
1. Are there known production incidents related to cleanup failures?
2. What is the actual cleanup order guarantee in the orchestrator? (source first, then transforms, then sinks?)
3. Are there performance concerns with cleanup duration under high plugin counts?

**Assumed But Not Verified:**
1. Transform cleanup order doesn't matter (tests don't verify deterministic order)
2. Orchestrator._cleanup_transforms is only called once per run (tests don't verify idempotency)
3. Graph._sink_id_map and _transform_id_map mutations are safe in test context

---

## Poorly Constructed Tests

### Test: test_cleanup_handles_missing_close_method (line 243)

**Issue:** Test name misleads - validates default close() implementation, not "missing" close()

**Evidence:**
```python
def test_cleanup_handles_missing_close_method(self) -> None:
    """Cleanup should handle transforms that use default close() method.

    BaseTransform provides a default no-op close() method, so transforms
    that don't override it still satisfy the protocol.
    """
```

The test constructs a transform that uses BaseTransform's default close() (a no-op). This is not "missing" - it's a valid implementation. The name suggests the orchestrator handles plugins without close() methods, which would be a protocol violation.

**Fix:**
- Rename to `test_cleanup_calls_default_close_implementation`
- Update docstring to remove "missing close() method" language
- Add comment: "Verifies orchestrator calls close() even when implementation is no-op"

**Priority:** P2 (confusing but functionally correct)

---

### Test: test_cleanup_continues_if_one_close_fails (line 282)

**Issue:** Assertion incomplete - doesn't verify cleanup failure was logged

**Evidence:**
```python
result = orchestrator.run(config, graph=_build_test_graph(config))

assert result.status == "completed"
# Both close() methods should have been called
assert transform_1.close_called, "failing transform's close() was not called"
assert transform_2.close_called, "second transform's close() was not called despite first failing"
```

Test verifies close() was called but doesn't verify the orchestrator logged the cleanup failure. According to orchestrator.py:202-221, failures should be logged via structlog. The test doesn't capture or verify this log output.

**Fix:**
Add log assertion:
```python
import structlog
from structlog.testing import LogCapture

def test_cleanup_continues_if_one_close_fails(self, caplog) -> None:
    # ... existing setup ...

    with caplog.at_level(logging.WARNING):
        result = orchestrator.run(config, graph=_build_test_graph(config))

    assert result.status == "completed"
    assert transform_1.close_called
    assert transform_2.close_called

    # Verify failure was logged
    assert any("Transform cleanup failed" in record.message for record in caplog.records)
    assert any("failing_close" in record.message for record in caplog.records)
```

**Priority:** P1 (best-effort cleanup contract not fully validated)

---

### Test: All tests (graph construction pattern)

**Issue:** Tests use brittle graph construction helper that directly mutates private graph state

**Evidence:**
```python
# From _build_test_graph (lines 61-78):
graph._sink_id_map = sink_ids
graph._transform_id_map = transform_ids
graph._route_resolution_map = route_resolution_map
graph._output_sink = ...
```

Tests bypass ExecutionGraph's public API and directly write to private fields (fields prefixed with `_`). This violates encapsulation and creates maintenance burden - if ExecutionGraph internals change, all cleanup tests break.

**Fix:**
1. Use ExecutionGraph.from_config() if available, OR
2. Add ExecutionGraph.for_testing() builder method that provides clean API for test graph construction, OR
3. Document why direct mutation is acceptable in test context (e.g., performance, simplicity) and add comment explaining this is test-only pattern

**Priority:** P2 (maintenance burden, not a correctness issue)

---

## Misclassified Tests

### All tests in this file (unit vs integration)

**Issue:** Tests are integration tests disguised as unit tests

**Evidence:**
- Tests instantiate Orchestrator with real LandscapeDB
- Tests run full orchestrator.run() lifecycle
- Tests verify end-to-end cleanup behavior across multiple subsystems (source, transforms, sinks, database)

**Current Classification:** Unit test (file location: `tests/engine/test_orchestrator_cleanup.py` alongside unit tests)

**Correct Classification:** Integration test - validates interaction between Orchestrator, LandscapeDB, RowProcessor, and plugins

**Fix:**
Move to `tests/integration/test_orchestrator_cleanup.py` OR rename file to `test_orchestrator_cleanup_integration.py` to signal this is not a fast unit test.

**Priority:** P2 (organizational clarity, affects test selection in CI)

---

## Infrastructure Gaps

### Gap 1: No source/sink cleanup tests

**Issue:** Tests only verify transform cleanup, not source/sink cleanup

**Evidence:**
According to orchestrator.py:1142-1146, the orchestrator calls close() on sources and sinks:
```python
# Close source and all sinks
config.source.close()
for sink in config.sinks.values():
    sink.close()
```

But test file has ZERO tests verifying:
1. source.close() is called on success
2. source.close() is called on failure
3. sink.close() is called on success
4. sink.close() is called on failure
5. source.close() failure doesn't prevent transform cleanup
6. sink.close() failure doesn't prevent transform cleanup

**Fix:**
Add tests:
```python
def test_source_closed_on_success(self) -> None:
    """Source.close() should be called after successful run."""

def test_source_closed_on_failure(self) -> None:
    """Source.close() should be called even if run fails."""

def test_sinks_closed_on_success(self) -> None:
    """All sinks should have close() called after successful run."""

def test_sinks_closed_on_failure(self) -> None:
    """All sinks should have close() called even if run fails."""

def test_source_close_failure_doesnt_prevent_transform_cleanup(self) -> None:
    """If source.close() raises, transforms should still be cleaned up."""

def test_sink_close_failure_doesnt_prevent_transform_cleanup(self) -> None:
    """If sink.close() raises, transforms should still be cleaned up."""
```

**Priority:** P0 (CRITICAL - production resource leak risk)

**Rationale:** File is named `test_orchestrator_cleanup.py` but only tests 1/3 of the cleanup surface area (transforms). Sources and sinks are equally important for resource management (file handles, database connections, HTTP clients).

---

### Gap 2: No cleanup ordering verification

**Issue:** Tests don't verify cleanup happens in correct order

**Evidence:**
The orchestrator has two cleanup locations:

1. **Transform cleanup** (orchestrator.py:588, called in finally block of run())
2. **Source/sink cleanup** (orchestrator.py:1142-1146, called in finally block of _execute_run())

The nesting structure suggests transforms are cleaned BEFORE sources/sinks:
```python
def run(...):
    try:
        _execute_run(...)  # <-- This finally block closes source/sinks
    finally:
        _cleanup_transforms(config)  # <-- This finally block closes transforms
```

**But:** None of the tests verify this ordering. If ordering matters (e.g., transforms depend on source/sink resources), the tests don't catch violations.

**Fix:**
Add test:
```python
def test_cleanup_order_transforms_before_source_and_sinks(self) -> None:
    """Transforms should be cleaned up before source and sinks.

    Rationale: Transforms may hold references to source/sink resources.
    Closing source/sink first could cause transform cleanup to fail.
    """
    cleanup_order = []

    class OrderTrackingTransform(TrackingTransform):
        def close(self) -> None:
            cleanup_order.append("transform")
            super().close()

    class OrderTrackingSource(ListSource):
        def close(self) -> None:
            cleanup_order.append("source")

    class OrderTrackingSink(CollectSink):
        def close(self) -> None:
            cleanup_order.append("sink")

    # Run pipeline
    orchestrator.run(config, graph=_build_test_graph(config))

    # Verify order: transforms closed BEFORE source and sinks
    assert cleanup_order == ["transform", "source", "sink"]
```

**Priority:** P1 (contract violation risk - ordering may be required but is untested)

---

### Gap 3: No exception suppression tests for source/sink cleanup

**Issue:** Tests verify transform cleanup uses best-effort exception handling, but don't test source/sink cleanup exception handling

**Evidence:**
From orchestrator.py:1142-1146:
```python
# Close source and all sinks
config.source.close()
for sink in config.sinks.values():
    sink.close()
```

**NO exception handling.** If source.close() raises, sinks are never closed. If first sink.close() raises, remaining sinks are never closed.

Compare to transform cleanup (orchestrator.py:212-221):
```python
for transform in config.transforms:
    try:
        transform.close()
    except Exception as e:
        logger.warning(...)  # Best-effort
```

**Fix:**
Add tests:
```python
def test_source_close_failure_doesnt_prevent_sink_cleanup(self) -> None:
    """If source.close() raises, sinks should still be cleaned up.

    EXPECTED TO FAIL: Current implementation has no exception handling
    for source.close(), so an exception will prevent sink cleanup.
    """
    class FailingCloseSource(ListSource):
        def close(self) -> None:
            raise RuntimeError("Source cleanup failed")

    source = FailingCloseSource([{"value": 1}])
    sink = CollectSink()

    # This should NOT raise, even though source.close() fails
    orchestrator.run(config, graph=_build_test_graph(config))

    # Sink should still be closed (not currently the case!)
    # This test will FAIL until orchestrator wraps source.close() in try/except

def test_first_sink_close_failure_doesnt_prevent_other_sinks(self) -> None:
    """If one sink.close() raises, other sinks should still be cleaned up."""
    # Similar pattern to above
```

**Priority:** P1 (production bug - partial cleanup on exception)

---

### Gap 4: No idempotency tests

**Issue:** Tests don't verify cleanup methods can't be called multiple times

**Evidence:**
Tests track close_call_count (line 164, 177, 213, 215) but don't verify the orchestrator PREVENTS multiple calls. They only verify it happened once in the success case.

**Missing scenarios:**
1. Can _cleanup_transforms() be called twice? (e.g., manual call + finally block)
2. Do plugins crash if close() called multiple times?
3. Does orchestrator protect against re-entrant cleanup?

**Fix:**
Add test:
```python
def test_cleanup_idempotent_multiple_calls_safe(self) -> None:
    """Calling _cleanup_transforms multiple times should be safe.

    Orchestrator uses finally blocks, which guarantee cleanup runs.
    But explicit cleanup calls might also occur (e.g., in exceptional paths).
    Plugins should tolerate multiple close() calls.
    """
    db = LandscapeDB.in_memory()
    transform = TrackingTransform()

    config = PipelineConfig(
        source=as_source(ListSource([{"value": 1}])),
        transforms=[transform],
        sinks={"default": as_sink(CollectSink())},
    )

    orchestrator = Orchestrator(db)
    orchestrator.run(config, graph=_build_test_graph(config))

    # First cleanup already happened via finally block
    assert transform.close_call_count == 1

    # Manually call cleanup again (simulates double-cleanup bug)
    orchestrator._cleanup_transforms(config)

    # Should NOT crash, but will increment counter
    assert transform.close_call_count == 2
    # NOTE: This reveals the orchestrator doesn't prevent double-cleanup.
    # Should it? That's a design decision, but test should document behavior.
```

**Priority:** P2 (defensive validation - prevents subtle double-cleanup bugs)

---

### Gap 5: No multi-sink cleanup tests

**Issue:** All tests use single sink (name: "default"). Production pipelines use multiple sinks.

**Evidence:**
```python
sinks={"default": as_sink(sink)}  # All tests use single sink
```

**Missing scenarios:**
1. Are all sinks closed when there are 3 sinks?
2. If sink 2/3 fails cleanup, does sink 3/3 still get cleaned?
3. Are sinks closed in deterministic order?

**Fix:**
Add test:
```python
def test_all_sinks_closed_when_multiple_sinks(self) -> None:
    """All sinks should be closed when pipeline has multiple sinks."""
    sink_1 = CollectSink()
    sink_2 = CollectSink()
    sink_3 = CollectSink()

    # Track which sinks were closed
    closed_sinks = []

    def track_close(original_close, sink_name):
        def wrapper():
            closed_sinks.append(sink_name)
            original_close()
        return wrapper

    sink_1.close = track_close(sink_1.close, "sink_1")
    sink_2.close = track_close(sink_2.close, "sink_2")
    sink_3.close = track_close(sink_3.close, "sink_3")

    config = PipelineConfig(
        source=as_source(ListSource([{"value": 1}])),
        transforms=[],
        sinks={
            "sink_1": as_sink(sink_1),
            "sink_2": as_sink(sink_2),
            "sink_3": as_sink(sink_3),
        },
    )

    orchestrator.run(config, graph=_build_test_graph(config))

    # All sinks should be closed
    assert set(closed_sinks) == {"sink_1", "sink_2", "sink_3"}
```

**Priority:** P2 (production pattern validation)

---

### Gap 6: No cleanup-during-exception tests

**Issue:** Tests verify cleanup happens after exceptions, but don't verify cleanup during ACTIVE exception handling

**Evidence:**
test_transforms_closed_on_failure (line 217) verifies cleanup after source failure:
```python
with pytest.raises(RuntimeError, match="Source failed intentionally"):
    orchestrator.run(config, graph=_build_test_graph(config))

# After exception, verify cleanup
assert transform_1.close_called
```

**Missing:** What if cleanup itself raises DURING the active exception? Python's exception chaining behavior applies here.

**Fix:**
Add test:
```python
def test_cleanup_exception_during_active_exception(self) -> None:
    """If cleanup raises during active exception, both are preserved.

    When source fails AND transform.close() fails, the orchestrator should:
    1. Let the SOURCE exception propagate (it's the root cause)
    2. Log (but not raise) the cleanup exception

    This prevents cleanup failures from masking the real error.
    """
    class FailingSource(ListSource):
        def load(self, ctx: Any) -> Any:
            raise RuntimeError("SOURCE FAILED")

    class FailingCloseTransform(TrackingTransform):
        def close(self) -> None:
            super().close()
            raise RuntimeError("CLEANUP FAILED")

    config = PipelineConfig(
        source=as_source(FailingSource([{"value": 1}])),
        transforms=[FailingCloseTransform()],
        sinks={"default": as_sink(CollectSink())},
    )

    # The SOURCE exception should propagate, not the cleanup exception
    with pytest.raises(RuntimeError, match="SOURCE FAILED"):
        orchestrator.run(config, graph=_build_test_graph(config))

    # Cleanup should have been attempted (and failed, but suppressed)
    # Log assertion would go here (see Gap #7)
```

**Priority:** P1 (exception masking risk - cleanup errors can hide root cause)

---

### Gap 7: No logging/observability tests

**Issue:** Tests don't verify cleanup events are observable

**Evidence:**
The orchestrator logs cleanup failures (orchestrator.py:217-221) but tests don't capture or verify these logs. Production operators need these logs for debugging resource leaks.

**Fix:**
Add structlog LogCapture fixture to all cleanup failure tests:
```python
from structlog.testing import LogCapture

@pytest.fixture
def log_capture():
    return LogCapture()

def test_cleanup_failure_logged(self, log_capture) -> None:
    # Configure structlog to write to log_capture
    import structlog
    structlog.configure(processors=[log_capture])

    # ... run test that causes cleanup failure ...

    # Verify log entry
    assert len(log_capture.entries) > 0
    assert any(
        entry["event"] == "Transform cleanup failed"
        and entry["transform"] == "failing_close"
        for entry in log_capture.entries
    )
```

**Priority:** P2 (observability validation)

---

### Gap 8: No plugin lifecycle state machine tests

**Issue:** Tests verify close() is called but don't verify it's called at the RIGHT TIME in the lifecycle

**Evidence:**
Plugin lifecycle per CLAUDE.md:
1. on_start(ctx) - called before processing
2. process(row, ctx) / load(ctx) / write(rows, ctx) - main work
3. on_complete(ctx) - called after processing
4. close() - called after on_complete

Tests verify close() is called but don't verify the ordering relative to on_complete().

**Fix:**
Add test:
```python
def test_cleanup_happens_after_on_complete(self) -> None:
    """close() should be called AFTER on_complete() in plugin lifecycle."""
    lifecycle_events = []

    class LifecycleTrackingTransform(TrackingTransform):
        def on_start(self, ctx: Any) -> None:
            lifecycle_events.append("on_start")

        def on_complete(self, ctx: Any) -> None:
            lifecycle_events.append("on_complete")

        def close(self) -> None:
            lifecycle_events.append("close")
            super().close()

    config = PipelineConfig(
        source=as_source(ListSource([{"value": 1}])),
        transforms=[LifecycleTrackingTransform()],
        sinks={"default": as_sink(CollectSink())},
    )

    orchestrator.run(config, graph=_build_test_graph(config))

    # Verify lifecycle order
    assert lifecycle_events == ["on_start", "on_complete", "close"]
```

**Priority:** P1 (contract verification - lifecycle ordering is part of plugin protocol)

---

## Positive Observations

**Well-structured test helpers:**
- `_build_test_graph()` centralizes graph construction logic (despite encapsulation concerns noted above)
- `TrackingTransform` provides clean assertion surface with `close_called` and `close_call_count`
- `FailingCloseTransform` properly extends TrackingTransform to preserve tracking behavior

**Good test naming:**
- Most test names clearly describe behavior being validated
- Docstrings explain the "why" behind tests, not just the "what"

**Exception safety coverage:**
- Tests verify cleanup happens on failure paths (test_transforms_closed_on_failure)
- Tests verify cleanup continues when one plugin fails (test_cleanup_continues_if_one_close_fails)

**Type annotations:**
- All test methods properly typed with `-> None`
- Helper methods have full type signatures

---

## Recommended Additions (Priority Ordered)

### P0 (Must Fix Before Release)
1. Add source cleanup tests (6 tests: success, failure, exception handling)
2. Add sink cleanup tests (6 tests: success, failure, multi-sink, exception handling)

### P1 (Should Fix Before Release)
1. Add cleanup ordering verification test
2. Add exception-during-exception test
3. Add plugin lifecycle state machine test
4. Add logging verification to cleanup failure tests
5. Fix misleading test name: test_cleanup_handles_missing_close_method

### P2 (Nice to Have)
1. Add idempotency tests
2. Add multi-sink cleanup tests
3. Add graph construction helper documentation or refactor to use public API
4. Consider moving to integration test directory

---

## Architecture Concerns (Findings for Orchestrator Implementation)

**CRITICAL BUG DETECTED:** Source/sink cleanup lacks exception handling

**Location:** orchestrator.py:1142-1146

**Evidence:**
```python
# Close source and all sinks
config.source.close()  # <-- If this raises, sinks never closed
for sink in config.sinks.values():
    sink.close()  # <-- If sink 1 raises, sink 2+ never closed
```

**Impact:** Production resource leaks when close() methods fail

**Fix Required:**
```python
# Close source and all sinks - best-effort cleanup
with suppress(Exception):
    config.source.close()
for sink in config.sinks.values():
    with suppress(Exception):
        sink.close()
```

This matches the transform cleanup pattern (orchestrator.py:212-221) and aligns with "best-effort cleanup" contract described in test docstrings.

**Test Gap:** None of the existing tests would catch this bug because they only test single-sink, success-path scenarios.

---

## Appendix: Test Coverage Gaps Summary

| Plugin Type | Success Path | Failure Path | Exception Handling | Ordering | Multi-Instance |
|-------------|--------------|--------------|-------------------|----------|----------------|
| Transform   | ✅ Tested    | ✅ Tested    | ✅ Tested         | ❌ Missing | ❌ Missing |
| Source      | ❌ Missing   | ❌ Missing   | ❌ Missing        | ❌ Missing | N/A (always 1) |
| Sink        | ❌ Missing   | ❌ Missing   | ❌ Missing        | ❌ Missing | ❌ Missing |

**Coverage Assessment:** ~30% of cleanup surface area tested. Transform cleanup is well-covered, but source/sink cleanup is completely untested despite being equally critical for production resilience.
