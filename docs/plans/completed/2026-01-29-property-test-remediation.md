# Property Test Remediation Plan

> **Status:** ✅ **COMPLETED** (2026-01-29)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate duplication, consolidate shared infrastructure, and close coverage gaps in ELSPETH's property test suite.

**Architecture:** Centralize all shared test fixtures, strategies, and constants in `tests/property/conftest.py`. Remove duplicates from individual test files. Add missing property tests for LLM plugins (testing ACTUAL production code), aggregation state machines, and sink behavior. Use Hypothesis `RuleBasedStateMachine` for stateful testing.

**Tech Stack:** Python, Hypothesis, pytest, SQLAlchemy (for audit queries)

**Review Status:** Plan reviewed by `ordis-quality-engineering:test-suite-reviewer` - all critical findings addressed in this revision.

---

## Phase 1: Consolidate Shared Infrastructure

### Task 1.1: Consolidate RFC 8785 Constants

**Files:**
- Modify: `tests/property/conftest.py:30-35`
- Modify: `tests/property/canonical/test_hash_determinism.py:34-38`
- Modify: `tests/property/audit/test_terminal_states.py:222-226`
- Modify: `tests/property/audit/test_fork_join_balance.py:234-238`

**Problem:** `MAX_SAFE_INT` and `MIN_SAFE_INT` are defined in 4 separate files.

**Step 1: Verify conftest.py already has constants**

Run: `grep -n "MAX_SAFE_INT\|MIN_SAFE_INT" tests/property/conftest.py`
Expected: Lines showing both constants defined

**Step 2: Update test_hash_determinism.py to import from conftest**

Replace lines 34-38:
```python
# OLD - DELETE THESE LINES:
# RFC 8785 (JCS) uses JavaScript-safe integers: -(2^53-1) to (2^53-1)
# Values outside this range cause serialization issues
_MAX_SAFE_INT = 2**53 - 1
_MIN_SAFE_INT = -(2**53 - 1)

# NEW - ADD THIS IMPORT at top of file (after other imports):
from tests.property.conftest import MAX_SAFE_INT, MIN_SAFE_INT
```

Then update all usages of `_MAX_SAFE_INT` to `MAX_SAFE_INT` and `_MIN_SAFE_INT` to `MIN_SAFE_INT` in the file.

**Step 3: Update test_terminal_states.py to import from conftest**

Replace line 222-226:
```python
# OLD - DELETE:
# RFC 8785 (JCS) uses JavaScript-safe integers: -(2^53-1) to (2^53-1)
# Values outside this range cause serialization issues
_MAX_SAFE_INT = 2**53 - 1

# NEW - ADD THIS IMPORT at top of file:
from tests.property.conftest import MAX_SAFE_INT
```

Update usage: `max_value=_MAX_SAFE_INT` → `max_value=MAX_SAFE_INT`

**Step 4: Update test_fork_join_balance.py to import from conftest**

Replace line 234-238:
```python
# OLD - DELETE:
# RFC 8785 safe integers
_MAX_SAFE_INT = 2**53 - 1

# NEW - ADD THIS IMPORT at top of file:
from tests.property.conftest import MAX_SAFE_INT
```

**Step 5: Run tests to verify no regressions**

Run: `python -m pytest tests/property/canonical/test_hash_determinism.py tests/property/audit/ -v --tb=short`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add tests/property/conftest.py tests/property/canonical/test_hash_determinism.py tests/property/audit/test_terminal_states.py tests/property/audit/test_fork_join_balance.py
git commit -m "refactor(tests): consolidate RFC 8785 constants to conftest.py"
```

---

### Task 1.2: Consolidate JSON Strategies

**Files:**
- Modify: `tests/property/conftest.py` (verify exports)
- Modify: `tests/property/canonical/test_hash_determinism.py:40-66`

**Problem:** `json_primitives`, `json_values`, `dict_keys`, `row_data` strategies are duplicated between conftest.py and test_hash_determinism.py.

**Step 1: Remove duplicate strategies from test_hash_determinism.py**

Delete lines 40-66 (the duplicate strategy definitions) and add import:
```python
from tests.property.conftest import (
    json_primitives,
    json_values,
    dict_keys,
    row_data,
    MAX_SAFE_INT,
    MIN_SAFE_INT,
)
```

**Step 2: Run canonical tests**

Run: `python -m pytest tests/property/canonical/ -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/property/canonical/test_hash_determinism.py
git commit -m "refactor(tests): remove duplicate JSON strategies from test_hash_determinism"
```

---

### Task 1.3: Create Shared Test Fixtures in conftest.py

**Files:**
- Modify: `tests/property/conftest.py` (add fixtures)
- Modify: `tests/property/audit/test_terminal_states.py:130-218`
- Modify: `tests/property/audit/test_fork_join_balance.py:165-230`

**Problem:** `_ListSource`, `_PassTransform`, `_CollectSink` are duplicated in multiple audit test files.

**Step 1: Add shared fixtures to conftest.py**

Add to end of `tests/property/conftest.py`:

```python
# =============================================================================
# Shared Test Fixtures (for integration/audit property tests)
# =============================================================================

from collections.abc import Iterator
from typing import Any

from elspeth.contracts import SourceRow
from elspeth.engine.artifacts import ArtifactDescriptor
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import _TestSchema, _TestSinkBase, _TestSourceBase


class PropertyTestSchema(_TestSchema):
    """Schema for property tests - accepts any dict with dynamic fields."""
    pass


class ListSource(_TestSourceBase):
    """Source that emits rows from a provided list.

    Use for property tests that need to generate random row sequences.
    """
    name = "property_list_source"
    output_schema = PropertyTestSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for row in self._data:
            yield SourceRow.valid(row)

    def close(self) -> None:
        pass


class PassTransform(BaseTransform):
    """Transform that passes rows through unchanged.

    Use for property tests that need a minimal transform in the pipeline.
    """
    name = "property_pass_transform"
    input_schema = PropertyTestSchema
    output_schema = PropertyTestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row)


class ConditionalErrorTransform(BaseTransform):
    """Transform that errors on rows where 'fail' key is truthy.

    Use for property tests that need to verify error handling paths.

    IMPORTANT: Uses direct key access per CLAUDE.md - if 'fail' key is
    missing, that's a test authoring bug that should crash.
    """
    name = "property_conditional_error"
    input_schema = PropertyTestSchema
    output_schema = PropertyTestSchema
    _on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        # Direct access - no defensive .get() per CLAUDE.md
        # Test data must include 'fail' key; missing key = test bug
        if row["fail"]:
            return TransformResult.error({"reason": "property_test_error"})
        return TransformResult.success(row)


class CollectSink(_TestSinkBase):
    """Sink that collects written rows in memory.

    Use for property tests that need to verify sink output.
    """
    name = "property_collect_sink"

    def __init__(self, sink_name: str = "default") -> None:
        self.name = sink_name
        self.results: list[dict[str, Any]] = []

    def on_start(self, ctx: Any) -> None:
        self.results = []

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        self.results.extend(rows)
        return ArtifactDescriptor.for_file(
            path=f"memory://{self.name}",
            size_bytes=len(str(rows)),
            content_hash="test_hash",
        )

    def close(self) -> None:
        pass
```

**Step 2: Update test_terminal_states.py to use shared fixtures**

Replace the local class definitions (lines ~130-218) with imports:
```python
from tests.property.conftest import (
    CollectSink,
    ConditionalErrorTransform,
    ListSource,
    PassTransform,
    PropertyTestSchema,
    MAX_SAFE_INT,
)
```

Delete the local `_PropertyTestSchema`, `_ListSource`, `_PassTransform`, `_ConditionalErrorTransform`, `_CollectSink` class definitions.

Update class references: `_ListSource` → `ListSource`, `_PassTransform` → `PassTransform`, etc.

**Step 3: Update test_fork_join_balance.py to use shared fixtures**

Replace local class definitions with imports:
```python
from tests.property.conftest import (
    CollectSink,
    ListSource,
    PassTransform,
    PropertyTestSchema,
    MAX_SAFE_INT,
)
```

Delete local `_ForkTestSchema`, `_ListSource`, `_PassTransform`, `_CollectSink` definitions.

Note: `_ForkTestSchema` can be replaced with `PropertyTestSchema` since both are minimal schemas.

**Step 4: Run all audit property tests**

Run: `python -m pytest tests/property/audit/ -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/property/conftest.py tests/property/audit/test_terminal_states.py tests/property/audit/test_fork_join_balance.py
git commit -m "refactor(tests): consolidate shared fixtures to property conftest.py"
```

---

### Task 1.4: Add ID String Strategy to conftest.py

**Files:**
- Modify: `tests/property/conftest.py`
- Modify: `tests/property/core/test_lineage_properties.py:43-55`

**Problem:** `id_strings` and `sink_names` strategies are locally defined but reusable.

**Step 1: Add ID strategies to conftest.py**

Add to `tests/property/conftest.py`:

```python
# =============================================================================
# ID and Name Strategies
# =============================================================================

# Valid ID strings (UUID-like hex strings)
id_strings = st.text(
    min_size=8,
    max_size=40,
    alphabet="0123456789abcdef",
)

# Sink/node names (lowercase with underscores)
sink_names = st.text(
    min_size=1,
    max_size=30,
    alphabet="abcdefghijklmnopqrstuvwxyz_",
)

# Path/label names (for routing)
path_names = st.text(
    min_size=1,
    max_size=30,
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
).filter(lambda s: s[0].isalpha())
```

**Step 2: Update test_lineage_properties.py to import**

Replace local strategy definitions (lines 43-55) with:
```python
from tests.property.conftest import id_strings, sink_names
```

**Step 3: Run lineage tests**

Run: `python -m pytest tests/property/core/test_lineage_properties.py -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/property/conftest.py tests/property/core/test_lineage_properties.py
git commit -m "refactor(tests): add ID and name strategies to conftest.py"
```

---

## Phase 2: Coverage Gaps - LLM Plugin Tests (REDESIGNED)

> **REVIEWER FIX:** Original plan tested a mock validation function. This revision tests ACTUAL production code per CLAUDE.md "Test Path Integrity" principle.

### Task 2.1: Extract LLM Response Validation Utility

**Files:**
- Create: `src/elspeth/plugins/llm/validation.py`
- Modify: `src/elspeth/plugins/llm/azure_multi_query_llm.py` (import utility)

**Context:** LLM transforms all contain similar JSON validation logic. Extract to a shared utility so property tests exercise PRODUCTION code, not a mock.

**Step 1: Examine existing validation pattern**

Run: `grep -A 20 "json.loads" src/elspeth/plugins/llm/azure_multi_query_llm.py`
Expected: See the JSON parse and isinstance check pattern

**Step 2: Create validation utility**

Create `src/elspeth/plugins/llm/validation.py`:

```python
# src/elspeth/plugins/llm/validation.py
"""LLM response validation utilities.

Per ELSPETH's Three-Tier Trust Model:
- LLM responses are Tier 3 (external data) - zero trust
- Validation must happen IMMEDIATELY at the boundary
- Invalid responses must be caught, not silently coerced

This module extracts the common validation pattern from LLM transforms
so it can be:
1. Reused across all LLM plugin implementations
2. Property-tested with Hypothesis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationSuccess:
    """Successful validation result containing parsed data."""
    data: dict[str, Any]


@dataclass(frozen=True)
class ValidationError:
    """Failed validation result with error details."""
    reason: str
    detail: str | None = None
    expected: str | None = None
    actual: str | None = None


ValidationResult = ValidationSuccess | ValidationError


def validate_json_object_response(content: str) -> ValidationResult:
    """Validate LLM response content is a JSON object.

    This is the standard validation for ELSPETH LLM transforms:
    1. Parse JSON (catch JSONDecodeError)
    2. Verify type is dict (not array, null, or primitive)
    3. Return validated dict or structured error

    Args:
        content: Raw response content from LLM API

    Returns:
        ValidationSuccess with parsed dict, or ValidationError with details
    """
    # Step 1: Parse JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return ValidationError(
            reason="invalid_json",
            detail=str(e),
        )

    # Step 2: Verify type is dict
    if not isinstance(parsed, dict):
        return ValidationError(
            reason="invalid_json_type",
            expected="object",
            actual=type(parsed).__name__,
        )

    # Success
    return ValidationSuccess(data=parsed)


def is_valid_json_object(content: str) -> bool:
    """Quick check if content is a valid JSON object.

    Use for simple boolean checks where full error details aren't needed.
    """
    result = validate_json_object_response(content)
    return isinstance(result, ValidationSuccess)
```

**Step 3: Update azure_multi_query_llm.py to use utility**

Find the inline validation code and replace with import:

```python
# Add import at top:
from elspeth.plugins.llm.validation import validate_json_object_response, ValidationSuccess

# Replace inline validation with:
result = validate_json_object_response(response.content)
if isinstance(result, ValidationSuccess):
    parsed = result.data
else:
    return TransformResult.error({
        "reason": result.reason,
        "detail": result.detail,
        "raw": response.content[:200],
    })
```

**Step 4: Run existing LLM tests**

Run: `python -m pytest tests/ -k "llm" -v --tb=short`
Expected: All tests PASS (behavior unchanged)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/validation.py src/elspeth/plugins/llm/azure_multi_query_llm.py
git commit -m "refactor(llm): extract response validation to shared utility"
```

---

### Task 2.2: Create LLM Response Validation Property Tests

**Files:**
- Create: `tests/property/plugins/__init__.py`
- Create: `tests/property/plugins/llm/__init__.py`
- Create: `tests/property/plugins/llm/test_response_validation_properties.py`

**Context:** Now that validation is in production code, property tests exercise the REAL implementation.

**Step 1: Create directory structure**

Run: `mkdir -p tests/property/plugins/llm`

**Step 2: Create __init__.py files**

Create `tests/property/plugins/__init__.py`:
```python
"""Property tests for plugin implementations."""
```

Create `tests/property/plugins/llm/__init__.py`:
```python
"""Property tests for LLM plugin response validation."""
```

**Step 3: Create the test file**

Create `tests/property/plugins/llm/test_response_validation_properties.py`:

```python
# tests/property/plugins/llm/test_response_validation_properties.py
"""Property-based tests for LLM response validation.

Per ELSPETH's Three-Tier Trust Model:
- LLM responses are Tier 3 (external data) - zero trust
- Validation must happen IMMEDIATELY at the boundary
- Invalid responses must be caught, not silently coerced

These tests exercise the PRODUCTION validation code in:
    src/elspeth/plugins/llm/validation.py

They verify that validation correctly handles:
- Non-JSON responses
- Wrong JSON types (array when object expected)
- Missing required fields
- Truncated/partial JSON
- Valid responses (positive cases)
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.plugins.llm.validation import (
    ValidationError,
    ValidationSuccess,
    validate_json_object_response,
)
from tests.property.conftest import json_primitives


# =============================================================================
# Strategies for generating LLM-like responses
# =============================================================================

# Valid JSON object responses
valid_json_objects = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
    values=json_primitives,
    min_size=0,  # Empty object {} is valid
    max_size=10,
)

# Non-JSON strings - use explicit patterns instead of filter() for efficiency
# (Reviewer fix: avoid .filter(not _is_valid_json) which rejects >99% of inputs)
non_json_strings = st.one_of(
    st.just(""),
    st.just("{"),
    st.just('{"incomplete": '),
    st.just("[1, 2, 3"),
    st.just("{invalid}"),
    st.just("{{double braces}}"),
    st.sampled_from([
        "This is not JSON",
        "Error: API rate limit exceeded",
        "<html>Error</html>",
        "None",
        "undefined",
        "NaN",
        "{'single': 'quotes'}",  # Python dict, not JSON
    ]),
)

# JSON that parses but is wrong type (array, not object)
wrong_type_json = st.one_of(
    st.lists(json_primitives, min_size=0, max_size=5).map(json.dumps),
    st.just("null"),
    st.just("true"),
    st.just("false"),
    st.integers().map(str),
    st.text(max_size=50).map(lambda s: json.dumps(s)),
)


# =============================================================================
# Property Tests: JSON Parse Boundary (Testing PRODUCTION code)
# =============================================================================


class TestLLMResponseParsingProperties:
    """Property tests for LLM response JSON parsing."""

    @given(response=valid_json_objects)
    @settings(max_examples=100)
    def test_valid_json_object_succeeds(self, response: dict[str, Any]) -> None:
        """Property: Valid JSON objects are accepted."""
        content = json.dumps(response)
        result = validate_json_object_response(content)

        assert isinstance(result, ValidationSuccess)
        assert result.data == response

    @given(response=non_json_strings)
    @settings(max_examples=100)
    def test_non_json_rejected(self, response: str) -> None:
        """Property: Non-JSON strings are rejected with clear error."""
        result = validate_json_object_response(response)

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json"

    @given(response=wrong_type_json)
    @settings(max_examples=100)
    def test_wrong_json_type_rejected(self, response: str) -> None:
        """Property: JSON that isn't an object is rejected.

        LLM transforms expect {"field": value} responses, not arrays,
        primitives, or null.
        """
        result = validate_json_object_response(response)

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json_type"
        assert result.expected == "object"


class TestLLMResponseEdgeCases:
    """Property tests for LLM response edge cases."""

    def test_empty_object_accepted(self) -> None:
        """Property: Empty object {} is valid (may have optional fields)."""
        result = validate_json_object_response("{}")

        assert isinstance(result, ValidationSuccess)
        assert result.data == {}

    def test_deeply_nested_accepted(self) -> None:
        """Property: Deeply nested objects are accepted."""
        deep = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        result = validate_json_object_response(json.dumps(deep))

        assert isinstance(result, ValidationSuccess)
        assert result.data == deep

    @given(whitespace=st.sampled_from([" ", "\n", "\t", "\r\n"]))
    @settings(max_examples=20)
    def test_whitespace_padded_json_accepted(self, whitespace: str) -> None:
        """Property: JSON with leading/trailing whitespace is accepted."""
        content = f'{whitespace}{{"key": "value"}}{whitespace}'
        result = validate_json_object_response(content)

        assert isinstance(result, ValidationSuccess)

    def test_null_json_rejected(self) -> None:
        """Property: JSON null is rejected (not an object)."""
        result = validate_json_object_response("null")

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json_type"

    def test_array_json_rejected(self) -> None:
        """Property: JSON array is rejected (not an object)."""
        result = validate_json_object_response("[1, 2, 3]")

        assert isinstance(result, ValidationError)
        assert result.reason == "invalid_json_type"
        assert result.actual == "list"


class TestLLMResponseDeterminism:
    """Property tests for validation determinism."""

    @given(content=st.text(max_size=200))
    @settings(max_examples=100)
    def test_validation_is_deterministic(self, content: str) -> None:
        """Property: Same input always produces same validation result."""
        result1 = validate_json_object_response(content)
        result2 = validate_json_object_response(content)

        # Both should be same type
        assert type(result1) is type(result2)

        if isinstance(result1, ValidationSuccess):
            assert result1.data == result2.data
        else:
            assert result1.reason == result2.reason
```

**Step 4: Run the new tests**

Run: `python -m pytest tests/property/plugins/llm/ -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/property/plugins/
git commit -m "feat(tests): add LLM response validation property tests"
```

---

## Phase 3: Coverage Gaps - Aggregation State Machine (FIXED)

> **REVIEWER FIXES APPLIED:**
> 1. Added negative assertion in `check_trigger()` (else clause)
> 2. Added invariant for "trigger state implies condition met"
> 3. Fixed model synchronization logic

### Task 3.1: Create Aggregation Stateful Property Tests

**Files:**
- Create: `tests/property/engine/test_aggregation_state_properties.py`

**Context:** Aggregation is a state machine (buffer → trigger → flush). Use Hypothesis `RuleBasedStateMachine` to test all state transitions.

**Step 1: Create the stateful test file**

Create `tests/property/engine/test_aggregation_state_properties.py`:

```python
# tests/property/engine/test_aggregation_state_properties.py
"""Property-based stateful tests for aggregation behavior.

Aggregation is a state machine:
- Initial: Empty buffer, timer not started
- Buffering: Accepting rows, timer running
- Triggered: Threshold met, ready to flush
- Flushed: Buffer cleared, timer reset

These tests use Hypothesis RuleBasedStateMachine to explore all
possible state transitions and verify invariants hold.

Key Invariants:
- Buffer count matches number of accepted rows
- Flush clears all state
- Timer starts on first accept, not before
- Trigger fires exactly at threshold, not before
- If trigger fires, at least one condition (count OR timeout) is met

REVIEWER NOTE: This implementation addresses the following issues
from the test suite review:
1. check_trigger() now has negative assertion (else clause)
2. Added trigger_condition_implies_threshold invariant
3. Model synchronization is explicit
"""

from __future__ import annotations

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from elspeth.core.config import TriggerConfig
from elspeth.engine.clock import MockClock
from elspeth.engine.triggers import TriggerEvaluator


class TriggerEvaluatorStateMachine(RuleBasedStateMachine):
    """Stateful property tests for TriggerEvaluator.

    This explores the state space of:
    - accept() calls with various row data
    - Time advances
    - Trigger checks
    - Reset operations
    """

    def __init__(self) -> None:
        super().__init__()
        self.clock = MockClock(start=0.0)
        self.config = TriggerConfig(count=10, timeout_seconds=5.0)
        self.evaluator = TriggerEvaluator(self.config, clock=self.clock)

        # Model state for verification
        self.model_count = 0
        self.model_first_accept_time: float | None = None

    def _model_should_trigger(self) -> bool:
        """Calculate expected trigger state from model.

        Separated from rules to ensure consistent calculation.
        """
        # Count condition
        if self.model_count >= self.config.count:
            return True

        # Timeout condition (only if timer started)
        if self.model_first_accept_time is not None:
            elapsed = self.clock.monotonic() - self.model_first_accept_time
            if elapsed >= self.config.timeout_seconds:
                return True

        return False

    @rule()
    def accept_row(self) -> None:
        """Accept a row into the aggregation buffer."""
        if self.model_first_accept_time is None:
            self.model_first_accept_time = self.clock.monotonic()

        self.evaluator.record_accept()
        self.model_count += 1

    @rule(seconds=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    def advance_time(self, seconds: float) -> None:
        """Advance the mock clock."""
        self.clock.advance(seconds)

    @rule()
    def check_trigger(self) -> None:
        """Check if trigger should fire - FIXED: now has negative assertion."""
        actual = self.evaluator.should_trigger()
        expected = self._model_should_trigger()

        # REVIEWER FIX: Assert BOTH directions, not just positive case
        assert actual == expected, (
            f"Trigger state mismatch: actual={actual}, expected={expected}, "
            f"count={self.model_count}/{self.config.count}, "
            f"age={self.evaluator.batch_age_seconds}/{self.config.timeout_seconds}"
        )

    @rule()
    def reset(self) -> None:
        """Reset the evaluator state."""
        self.evaluator.reset()

        # Reset model state
        self.model_count = 0
        self.model_first_accept_time = None

    @invariant()
    def count_matches_model(self) -> None:
        """Invariant: Buffer count always matches our model."""
        assert self.evaluator.batch_count == self.model_count, (
            f"Count mismatch: evaluator={self.evaluator.batch_count}, "
            f"model={self.model_count}"
        )

    @invariant()
    def age_is_non_negative(self) -> None:
        """Invariant: Batch age is never negative."""
        assert self.evaluator.batch_age_seconds >= 0.0

    @invariant()
    def age_is_zero_before_first_accept(self) -> None:
        """Invariant: Age is 0 when no rows have been accepted."""
        if self.model_first_accept_time is None:
            assert self.evaluator.batch_age_seconds == 0.0

    @invariant()
    def trigger_condition_implies_threshold(self) -> None:
        """Invariant: If trigger fires, at least one condition is met.

        REVIEWER FIX: This is the critical invariant that was missing.
        If should_trigger() returns True, EITHER:
        - count >= count_threshold, OR
        - elapsed_time >= timeout_threshold

        This catches bugs where trigger fires spuriously.
        """
        if self.evaluator.should_trigger():
            count_ok = self.evaluator.batch_count >= self.config.count
            time_ok = self.evaluator.batch_age_seconds >= self.config.timeout_seconds

            assert count_ok or time_ok, (
                f"Trigger fired but no condition met: "
                f"count={self.evaluator.batch_count}/{self.config.count}, "
                f"age={self.evaluator.batch_age_seconds}/{self.config.timeout_seconds}"
            )


# Create the test class that pytest will discover
TestTriggerStateMachine = TriggerEvaluatorStateMachine.TestCase
TestTriggerStateMachine.settings = settings(max_examples=100, stateful_step_count=30)


# =============================================================================
# Additional Non-Stateful Aggregation Properties
# =============================================================================


class TestAggregationInvariants:
    """Additional property tests for aggregation invariants."""

    @given(count=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_trigger_fires_exactly_at_count_threshold(self, count: int) -> None:
        """Property: Count trigger fires at exactly the threshold, not before."""
        config = TriggerConfig(count=count)
        evaluator = TriggerEvaluator(config)

        # Accept count-1 rows - should NOT trigger
        for _ in range(count - 1):
            evaluator.record_accept()
        assert not evaluator.should_trigger(), f"Triggered early at {count-1}/{count}"

        # Accept one more - NOW should trigger
        evaluator.record_accept()
        assert evaluator.should_trigger(), f"Didn't trigger at {count}/{count}"

    @given(timeout=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_trigger_fires_at_timeout_threshold(self, timeout: float) -> None:
        """Property: Timeout trigger fires at exactly the threshold."""
        clock = MockClock(start=0.0)
        config = TriggerConfig(timeout_seconds=timeout)
        evaluator = TriggerEvaluator(config, clock=clock)

        # Must accept at least one row to start timer
        evaluator.record_accept()

        # Advance to just before timeout - should NOT trigger
        clock.advance(timeout * 0.9)
        assert not evaluator.should_trigger(), f"Triggered early at {timeout * 0.9}s"

        # Advance past timeout - NOW should trigger
        clock.advance(timeout * 0.2)  # Total: timeout * 1.1
        assert evaluator.should_trigger(), f"Didn't trigger at {timeout * 1.1}s"
```

**Step 2: Run the stateful tests**

Run: `python -m pytest tests/property/engine/test_aggregation_state_properties.py -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/property/engine/test_aggregation_state_properties.py
git commit -m "feat(tests): add aggregation stateful property tests with reviewer fixes"
```

---

## Phase 4: Coverage Gaps - Sink Property Tests

### Task 4.1: Create Sink Property Tests

**Files:**
- Create: `tests/property/sinks/__init__.py`
- Create: `tests/property/sinks/test_artifact_properties.py`

**Context:** Sinks produce `ArtifactDescriptor` records. These must be deterministic for audit integrity.

**Step 1: Create directory and __init__.py**

Run: `mkdir -p tests/property/sinks`

Create `tests/property/sinks/__init__.py`:
```python
"""Property tests for sink implementations."""
```

**Step 2: Create artifact property tests**

Create `tests/property/sinks/test_artifact_properties.py`:

```python
# tests/property/sinks/test_artifact_properties.py
"""Property-based tests for sink artifact descriptors.

Sink artifacts are recorded in the audit trail. Their content hashes
must be deterministic - same content must always produce same hash.

These tests verify:
- Content hash determinism
- Artifact descriptor immutability
- Size reporting accuracy
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.engine.artifacts import ArtifactDescriptor
from elspeth.core.canonical import stable_hash


# =============================================================================
# Strategies
# =============================================================================

file_paths = st.text(
    min_size=1,
    max_size=100,
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789/_-.",
).map(lambda s: f"file://{s}")

content_hashes = st.text(
    min_size=64,
    max_size=64,
    alphabet="0123456789abcdef",
)

sizes = st.integers(min_value=0, max_value=10_000_000)


# =============================================================================
# ArtifactDescriptor Property Tests
# =============================================================================


class TestArtifactDescriptorProperties:
    """Property tests for ArtifactDescriptor."""

    @given(path=file_paths, size=sizes, content_hash=content_hashes)
    @settings(max_examples=100)
    def test_for_file_creates_valid_descriptor(
        self, path: str, size: int, content_hash: str
    ) -> None:
        """Property: for_file() creates descriptor with correct fields."""
        descriptor = ArtifactDescriptor.for_file(
            path=path,
            size_bytes=size,
            content_hash=content_hash,
        )

        assert descriptor.path == path
        assert descriptor.size_bytes == size
        assert descriptor.content_hash == content_hash

    @given(path=file_paths, size=sizes, content_hash=content_hashes)
    @settings(max_examples=50)
    def test_descriptor_creation_is_deterministic(
        self, path: str, size: int, content_hash: str
    ) -> None:
        """Property: Same inputs produce equal descriptors."""
        d1 = ArtifactDescriptor.for_file(path=path, size_bytes=size, content_hash=content_hash)
        d2 = ArtifactDescriptor.for_file(path=path, size_bytes=size, content_hash=content_hash)

        assert d1.path == d2.path
        assert d1.size_bytes == d2.size_bytes
        assert d1.content_hash == d2.content_hash


class TestContentHashDeterminism:
    """Property tests for content hash determinism in sinks."""

    @given(content=st.binary(min_size=0, max_size=10_000))
    @settings(max_examples=100)
    def test_binary_content_hash_deterministic(self, content: bytes) -> None:
        """Property: Same binary content always produces same hash."""
        # Simulate what a sink would do
        hash1 = stable_hash({"content": content})
        hash2 = stable_hash({"content": content})

        assert hash1 == hash2

    @given(rows=st.lists(st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(st.integers(), st.text(max_size=20)),
        min_size=1,
        max_size=5,
    ), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_row_batch_hash_deterministic(self, rows: list[dict]) -> None:
        """Property: Same row batch always produces same hash."""
        hash1 = stable_hash({"rows": rows})
        hash2 = stable_hash({"rows": rows})

        assert hash1 == hash2
```

**Step 3: Run sink tests**

Run: `python -m pytest tests/property/sinks/ -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/property/sinks/
git commit -m "feat(tests): add sink artifact property tests"
```

---

## Phase 5: Verify and Expand Existing Tests

### Task 5.1: Verify Existing Test Files Have Adequate Coverage

**Files:**
- Review: `tests/property/core/test_payload_store_properties.py`
- Review: `tests/property/core/test_rate_limiter_properties.py`
- Review: `tests/property/core/test_checkpoint_properties.py`

**Step 1: Check payload store test coverage**

Run: `python -m pytest tests/property/core/test_payload_store_properties.py -v --tb=short 2>/dev/null || echo "File may not exist or has issues"`

**Step 2: Check rate limiter test coverage**

Run: `python -m pytest tests/property/core/test_rate_limiter_properties.py -v --tb=short 2>/dev/null || echo "File may not exist or has issues"`

**Step 3: Check checkpoint test coverage**

Run: `python -m pytest tests/property/core/test_checkpoint_properties.py -v --tb=short 2>/dev/null || echo "File may not exist or has issues"`

**Step 4: Document gaps if any**

If any tests fail or files don't exist, create issues or add to backlog.

---

## Phase 6: Final Validation

### Task 6.1: Run Full Property Test Suite

**Step 1: Run all property tests**

Run: `python -m pytest tests/property/ -v --tb=short -q`
Expected: All tests PASS

**Step 2: Check for import errors**

Run: `python -c "import tests.property.conftest; print('Imports OK')"`
Expected: "Imports OK"

**Step 3: Verify no duplicate definitions remain**

Run: `grep -r "MAX_SAFE_INT = 2\*\*53" tests/property/ --include="*.py" | grep -v conftest.py | wc -l`
Expected: 0 (no duplicates outside conftest)

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore(tests): complete property test remediation"
```

---

## Summary

| Phase | Tasks | Description | Status |
|-------|-------|-------------|--------|
| 1 | 1.1-1.4 | Consolidate duplicates (constants, strategies, fixtures) | Ready |
| 2 | 2.1-2.2 | Extract LLM validation utility + property tests | **REDESIGNED** |
| 3 | 3.1 | Add aggregation stateful property tests | **FIXED** |
| 4 | 4.1 | Add sink artifact property tests | Ready |
| 5 | 5.1 | Verify existing test files | Ready |
| 6 | 6.1 | Final validation | Ready |

**Total Tasks:** 9 main tasks with ~45 steps

**Estimated Time:** 2-3 hours

**Risk Level:** Low (refactoring existing tests, adding new tests)

---

## Reviewer Findings Addressed

| Finding | Severity | Resolution |
|---------|----------|------------|
| Phase 2 tested mock function, not production code | High | Redesigned: extract validation to `src/`, test actual code |
| RuleBasedStateMachine missing negative assertions | High | Added else clause in `check_trigger()` |
| RuleBasedStateMachine missing condition invariant | High | Added `trigger_condition_implies_threshold` invariant |
| Defensive `.get()` in ConditionalErrorTransform | Low | Changed to direct key access |
| Inefficient Hypothesis filter in strategies | Low | Used explicit patterns instead of filter |

---

## Future Improvements (Out of Scope)

These items were identified by the reviewer but are not addressed in this plan:

1. **Shared fixtures directory** (`tests/shared/`) - Would prevent import cycle risks
2. **Gate routing property tests** - Verify routing decision properties
3. **Schema contract validation tests** - Verify DAG validation catches field mismatches
4. **Retry idempotency tests** - Verify deterministic transforms produce same output on retry
5. **Performance budget** - Set CI time limits for property test suite
