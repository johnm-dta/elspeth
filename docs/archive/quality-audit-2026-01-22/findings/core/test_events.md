# Test Quality Review: test_events.py

## Summary

The test suite for `core/events.py` is well-structured with good coverage of basic scenarios and appropriate adherence to the "let it crash" philosophy for handler bugs. However, it has significant gaps in mutation testing (frozen dataclass immutability verification), lacks property-based testing for event bus invariants, and misses critical edge cases around thread safety assumptions and handler state isolation.

---

## Poorly Constructed Tests

### Test: test_handler_ordering (line 66)
**Issue**: Test assumes subscription order is guaranteed but implementation uses dict iteration, which is only insertion-ordered in Python 3.7+. No explicit contract verification or Python version check.

**Evidence**:
```python
def test_handler_ordering(self) -> None:
    """Test handlers are called in subscription order."""
    # Implementation uses dict iteration - relies on Python 3.7+ insertion order
```

**Fix**: Add comment documenting Python 3.7+ requirement OR use explicit OrderedDict OR add test that verifies this is a guaranteed contract (if intentional).

**Priority**: P2 (works in practice on Python 3.7+, but contract is undocumented)

---

### Test: test_subscribe_same_handler_multiple_times (line 199)
**Issue**: Test documents "calls it multiple times" behavior but doesn't verify if this is intentional or a bug. Most event bus implementations deduplicate subscriptions.

**Evidence**:
```python
def test_subscribe_same_handler_multiple_times(self) -> None:
    """Test subscribing same handler multiple times calls it multiple times."""
    # Is this INTENDED behavior or a BUG?
    assert call_count == 2
```

**Fix**: Either:
1. Document in implementation that duplicate subscriptions are intentional (and why), OR
2. Fix implementation to deduplicate handlers, OR
3. Add unsubscribe mechanism to allow intentional multiple subscriptions

**Priority**: P1 (ambiguous contract - could cause production bugs)

---

### Test: test_handler_can_emit_events (line 217)
**Issue**: Tests re-entrancy but only one level deep. Doesn't verify protection against infinite recursion (e.g., handler1 emits Event1, handler for Event1 emits Event1 again).

**Evidence**:
```python
def handler1(event: TestEvent) -> None:
    received.append(f"handler1:{event.value}")
    bus.emit(AnotherEvent(count=42))  # Only emits DIFFERENT event type
```

**Fix**: Add test for circular event emissions (handler emits same event type) to verify intended behavior (crash? max depth? silent drop?).

**Priority**: P1 (re-entrancy protection is undefined)

---

## Missing Critical Tests

### Missing: Immutability verification for frozen dataclasses
**Issue**: Events are `@dataclass(frozen=True)` but no tests verify the frozen contract is enforced. This is critical for audit integrity - mutation of events after emission would violate traceability.

**Evidence**:
```python
@dataclass(frozen=True)
class TestEvent:
    """Test event for EventBus tests."""
    value: str
# NO TESTS that verify: event.value = "mutated" raises FrozenInstanceError
```

**Fix**: Add test class `TestEventImmutability`:
```python
def test_event_immutability_enforced(self) -> None:
    """Test frozen events cannot be mutated after creation."""
    event = TestEvent(value="original")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.value = "mutated"  # type: ignore[misc]

def test_handler_cannot_mutate_event(self) -> None:
    """Test handlers receive immutable events."""
    bus = EventBus()

    def mutating_handler(event: TestEvent) -> None:
        event.value = "mutated"  # type: ignore[misc]

    bus.subscribe(TestEvent, mutating_handler)

    with pytest.raises(dataclasses.FrozenInstanceError):
        bus.emit(TestEvent(value="original"))
```

**Priority**: P0 (CLAUDE.md: frozen dataclasses for immutability - audit integrity depends on this)

---

### Missing: Thread safety assumptions documented
**Issue**: EventBus uses plain `dict` and `list` which are not thread-safe for concurrent mutations. No tests or documentation clarify if concurrent `subscribe()` or `emit()` calls are supported.

**Evidence**: Implementation has no locks, and CLAUDE.md mentions OpenTelemetry spans (which often run in separate threads).

**Fix**: Either:
1. Document "single-threaded only" contract in docstring and add test verifying behavior is undefined for concurrent access, OR
2. Add thread-safety with locks and test concurrent subscribe/emit

**Priority**: P1 (production risk if orchestrator uses threads)

---

### Missing: Handler state isolation verification
**Issue**: No test verifies that handlers don't share mutable state through closure capture, which could violate "no shared mutable state" principle from CLAUDE.md.

**Evidence**: CLAUDE.md section on "Shared Mutable State" in test quality anti-patterns.

**Fix**: Add test demonstrating handlers with captured state are isolated:
```python
def test_handlers_dont_share_mutable_state(self) -> None:
    """Test each handler invocation receives same event but independent state."""
    bus = EventBus()

    def make_handler(storage: list[str]) -> Callable[[TestEvent], None]:
        def handler(event: TestEvent) -> None:
            storage.append(event.value)
        return handler

    storage1: list[str] = []
    storage2: list[str] = []

    bus.subscribe(TestEvent, make_handler(storage1))
    bus.subscribe(TestEvent, make_handler(storage2))

    bus.emit(TestEvent(value="shared"))

    # Both handlers should receive event, but have independent storage
    assert storage1 == ["shared"]
    assert storage2 == ["shared"]
    assert storage1 is not storage2
```

**Priority**: P2 (good practice, aligns with CLAUDE.md principles)

---

### Missing: Empty event type edge case
**Issue**: No test for subscribing to events with no fields (e.g., `@dataclass(frozen=True) class Ping: pass`). This is a valid use case for signaling.

**Fix**: Add test with minimal event type to verify EventBus doesn't assume events have fields.

**Priority**: P3 (edge case, unlikely to fail)

---

### Missing: EventBus reset/cleanup test
**Issue**: No test for resetting EventBus state (e.g., clearing all subscriptions). This may be needed for test isolation or runtime reconfiguration.

**Fix**: Either:
1. Add `clear()` method and test it, OR
2. Document "EventBus is append-only, create new instance to reset" in docstring

**Priority**: P2 (affects test isolation best practices)

---

## Misclassified Tests

### Test Class: TestEventBusProtocol (line 172)
**Issue**: These are not runtime tests - they verify static type checking. Running at runtime doesn't prove mypy compliance.

**Evidence**:
```python
def test_eventbus_satisfies_protocol(self) -> None:
    """Test EventBus satisfies EventBusProtocol."""
    def accepts_protocol(bus: EventBusProtocol) -> None:
        bus.subscribe(TestEvent, lambda e: None)
        bus.emit(TestEvent(value="test"))

    # Should not raise type errors  <-- This is a TYPE check, not a runtime check
    accepts_protocol(EventBus())
```

**Fix**: Move to integration tests that run `mypy` on example code, OR remove (redundant with mypy CI checks), OR convert to explicit protocol verification:
```python
def test_eventbus_implements_protocol_methods(self) -> None:
    """Test EventBus has required protocol methods."""
    bus = EventBus()
    assert hasattr(bus, 'subscribe')
    assert hasattr(bus, 'emit')
    assert callable(bus.subscribe)
    assert callable(bus.emit)
```

**Priority**: P2 (test provides no runtime value, only documentation)

---

## Infrastructure Gaps

### Gap: No fixtures for common event bus setup
**Issue**: Each test creates `EventBus()` manually. Repeated setup increases maintenance burden if EventBus constructor changes.

**Evidence**: `bus = EventBus()` appears in every test method.

**Fix**: Add pytest fixture in `conftest.py` or test file:
```python
@pytest.fixture
def event_bus() -> EventBus:
    """Fresh EventBus instance for each test."""
    return EventBus()

def test_subscribe_and_emit(self, event_bus: EventBus) -> None:
    # Use fixture instead of manual creation
```

**Priority**: P3 (nice-to-have, current pattern is acceptable)

---

### Gap: No property-based testing for event bus invariants
**Issue**: CLAUDE.md explicitly calls out Hypothesis for property testing, and event buses have clear invariants (order preservation, no lost events, type isolation). Current tests only check specific examples.

**Evidence**: CLAUDE.md states "Property Testing | Hypothesis | Manual edge-case hunting". EventBus has properties like:
- "All subscribed handlers receive event" (no drops)
- "Handlers called in subscription order"
- "Different event types never cross-contaminate"

**Fix**: Add property tests using Hypothesis:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1), min_size=1, max_size=100))
def test_all_handlers_receive_event_property(self, values: list[str]) -> None:
    """Property: Every subscribed handler receives emitted event."""
    bus = EventBus()
    received: dict[int, list[str]] = {i: [] for i in range(len(values))}

    for i, expected in enumerate(values):
        bus.subscribe(TestEvent, lambda e, idx=i: received[idx].append(e.value))

    event = TestEvent(value="test")
    bus.emit(event)

    # PROPERTY: All handlers called exactly once
    for handler_events in received.values():
        assert len(handler_events) == 1
        assert handler_events[0] == "test"

@given(st.lists(st.integers(), min_size=1, max_size=50))
def test_handler_ordering_preserved_property(self, call_indices: list[int]) -> None:
    """Property: Handlers always called in subscription order."""
    bus = EventBus()
    call_order: list[int] = []

    for idx in call_indices:
        bus.subscribe(TestEvent, lambda e, i=idx: call_order.append(i))

    bus.emit(TestEvent(value="test"))

    # PROPERTY: Call order matches subscription order
    assert call_order == call_indices
```

**Priority**: P1 (CLAUDE.md mandates property testing, and event bus has testable properties)

---

### Gap: No testing with dataclasses that have complex fields
**Issue**: Test events only use primitives (`str`, `int`). No test with nested dataclasses, lists, dicts, or types requiring normalization.

**Evidence**:
```python
@dataclass(frozen=True)
class TestEvent:
    value: str  # Only primitive

@dataclass(frozen=True)
class AnotherEvent:
    count: int  # Only primitive
```

**Fix**: Add event types with complex fields and verify event bus handles them:
```python
@dataclass(frozen=True)
class ComplexEvent:
    metadata: dict[str, Any]
    tags: list[str]
    nested: TestEvent | None = None

def test_complex_event_types(self) -> None:
    """Test EventBus handles events with nested/complex fields."""
    bus = EventBus()
    received: list[ComplexEvent] = []

    bus.subscribe(ComplexEvent, lambda e: received.append(e))

    event = ComplexEvent(
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
        nested=TestEvent(value="inner")
    )
    bus.emit(event)

    assert len(received) == 1
    assert received[0] is event  # Identity preserved
```

**Priority**: P2 (EventBus likely works, but untested assumption)

---

## Positive Observations

1. **Correct "let it crash" philosophy**: Handler exceptions propagate (test_handler_exception_propagates), aligning with CLAUDE.md "Plugin bugs are system bugs - crash immediately".

2. **Good test isolation**: Tests use independent event types (TestEvent, AnotherEvent) preventing cross-contamination.

3. **Clear test names and docstrings**: Every test has a descriptive docstring explaining intent.

4. **Fail-fast verification**: test_handler_exception_stops_subsequent_handlers correctly verifies exception halts processing.

5. **Protocol-based design tested**: Tests verify both EventBus and NullEventBus satisfy EventBusProtocol, aligning with CLAUDE.md's structural typing preference.

6. **Edge case coverage started**: TestEventBusEdgeCases class shows awareness of non-happy-path scenarios.

---

## SME Assessment Sections (Per SME Agent Protocol)

### Confidence Assessment
**Confidence Level**: High (85%)

**Justification**: Reviewed implementation code, test code, and CLAUDE.md standards. Event bus is a well-understood pattern with clear contracts. Identified issues are based on:
1. Direct comparison with CLAUDE.md principles (immutability, property testing)
2. Industry best practices for event bus testing (thread safety, re-entrancy)
3. Codebase conventions observed in other test files (Hypothesis usage)

**Uncertainty Areas**:
- Thread safety requirements (15% uncertain) - CLAUDE.md mentions OpenTelemetry spans but doesn't explicitly state if EventBus must be thread-safe
- Intended behavior for duplicate handler subscriptions - ambiguous if this is a feature or oversight

---

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Immutability not enforced** | Medium | CRITICAL | Add frozen dataclass mutation tests (P0) |
| **Concurrent access undefined** | Low-Medium | High | Document thread-safety contract or add locks (P1) |
| **Infinite recursion in handlers** | Low | High | Add re-entrancy depth test (P1) |
| **Duplicate subscriptions cause bugs** | Medium | Medium | Clarify contract and add deduplication if needed (P1) |
| **Property invariants broken on edge cases** | Low | Medium | Add Hypothesis-based property tests (P1) |

**Overall Risk**: Medium - Core functionality is tested, but critical audit integrity properties (immutability) are not verified.

---

### Information Gaps

1. **Thread-safety requirements**: Does orchestrator emit events from multiple threads? (affects priority of thread-safety tests)

2. **Production usage patterns**: Are there cases where duplicate handler subscriptions are intentional? (affects duplicate subscription fix)

3. **Event volume expectations**: Are high-frequency events expected? (affects need for performance tests)

4. **Unsubscribe mechanism**: Is runtime handler removal needed? (affects API completeness)

5. **Error recovery**: Should EventBus support retrying failed handlers or circuit-breaking? (affects exception handling tests)

---

### Caveats

1. **Static type checking limitation**: TestEventBusProtocol tests don't provide runtime guarantees - mypy in CI is the real verification.

2. **Python version assumption**: Handler ordering test assumes Python 3.7+ dict insertion order but doesn't document this requirement.

3. **Synchronous-only assumption**: All tests assume synchronous execution - async event buses would need completely different tests.

4. **No performance testing**: Review doesn't assess event bus performance under load (may need separate performance test suite).

5. **Framework-specific patterns**: Review assumes pytest conventions - different test frameworks would need different fixtures/assertions.

---

## Recommended Action Plan

**P0 (Fix before RC-1 release):**
1. Add immutability enforcement tests (frozen dataclass mutation protection)

**P1 (Fix in RC-1 bug burndown):**
1. Add property-based tests for event bus invariants (Hypothesis)
2. Document thread-safety contract or add thread-safety
3. Test/fix circular event emission behavior
4. Clarify duplicate subscription contract

**P2 (Post-RC-1 improvements):**
1. Add complex event type tests
2. Extract event_bus fixture
3. Add handler state isolation test
4. Document/test EventBus cleanup behavior

**P3 (Nice-to-have):**
1. Test empty event types
2. Remove redundant protocol tests (rely on mypy)
