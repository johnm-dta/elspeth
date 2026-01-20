# Pooled LLM Queries Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable parallel LLM API calls within a single transform while maintaining strict row order and gracefully handling capacity errors with AIMD throttling.

**Architecture:** Per-transform `PooledExecutor` manages a semaphore-controlled dispatch queue, AIMD throttle for adaptive rate control, and reorder buffer for strict output ordering. The executor wraps HTTP calls from existing `AuditedHTTPClient` and is synchronous from the caller's perspective.

**Tech Stack:** `concurrent.futures.ThreadPoolExecutor`, `threading.Semaphore`, existing `AuditedHTTPClient`, Pydantic config models, pytest/Hypothesis for testing.

**Design Document:** See `docs/plans/2026-01-20-pooled-llm-queries-design.md` for full specification.

---

## Task 1: AIMD Throttle State Machine

**Files:**
- Create: `src/elspeth/plugins/llm/aimd_throttle.py`
- Test: `tests/plugins/llm/test_aimd_throttle.py`

This is the core throttle algorithm: multiplicative decrease on capacity errors, additive increase on success.

### Step 1: Write failing test for throttle initialization

```python
# tests/plugins/llm/test_aimd_throttle.py
"""Tests for AIMD throttle state machine."""

import pytest

from elspeth.plugins.llm.aimd_throttle import AIMDThrottle, ThrottleConfig


class TestAIMDThrottleInit:
    """Test throttle initialization and defaults."""

    def test_default_config_values(self) -> None:
        """Verify sensible defaults are applied."""
        throttle = AIMDThrottle()

        assert throttle.current_delay_ms == 0
        assert throttle.config.min_dispatch_delay_ms == 0
        assert throttle.config.max_dispatch_delay_ms == 5000
        assert throttle.config.backoff_multiplier == 2.0
        assert throttle.config.recovery_step_ms == 50

    def test_custom_config(self) -> None:
        """Verify custom config is applied."""
        config = ThrottleConfig(
            min_dispatch_delay_ms=10,
            max_dispatch_delay_ms=1000,
            backoff_multiplier=3.0,
            recovery_step_ms=25,
        )
        throttle = AIMDThrottle(config)

        assert throttle.config.min_dispatch_delay_ms == 10
        assert throttle.config.max_dispatch_delay_ms == 1000
        assert throttle.config.backoff_multiplier == 3.0
        assert throttle.config.recovery_step_ms == 25
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleInit -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/llm/aimd_throttle.py
"""AIMD (Additive Increase, Multiplicative Decrease) throttle for LLM API calls.

Implements TCP-style congestion control:
- On capacity error: multiply delay (fast ramp down)
- On success: subtract fixed amount (slow ramp up)

This prevents "riding the edge" where you're constantly hitting capacity limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class ThrottleConfig:
    """Configuration for AIMD throttle behavior.

    Attributes:
        min_dispatch_delay_ms: Floor for delay between dispatches (default: 0)
        max_dispatch_delay_ms: Ceiling for delay (default: 5000)
        backoff_multiplier: Multiply delay on capacity error (default: 2.0)
        recovery_step_ms: Subtract from delay on success (default: 50)
    """

    min_dispatch_delay_ms: int = 0
    max_dispatch_delay_ms: int = 5000
    backoff_multiplier: float = 2.0
    recovery_step_ms: int = 50


class AIMDThrottle:
    """Thread-safe AIMD throttle state machine.

    Usage:
        throttle = AIMDThrottle()

        # Before dispatching request
        delay = throttle.current_delay_ms
        time.sleep(delay / 1000)

        # After request completes
        if is_capacity_error:
            throttle.on_capacity_error()
        else:
            throttle.on_success()
    """

    def __init__(self, config: ThrottleConfig | None = None) -> None:
        """Initialize throttle with optional config.

        Args:
            config: Throttle configuration (uses defaults if None)
        """
        self._config = config or ThrottleConfig()
        self._current_delay_ms: float = 0.0
        self._lock = Lock()

    @property
    def config(self) -> ThrottleConfig:
        """Get throttle configuration."""
        return self._config

    @property
    def current_delay_ms(self) -> float:
        """Get current delay in milliseconds (thread-safe)."""
        with self._lock:
            return self._current_delay_ms
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleInit -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/aimd_throttle.py tests/plugins/llm/test_aimd_throttle.py
git commit -m "feat(llm): add AIMD throttle config and initialization"
```

---

## Task 2: AIMD Throttle Backoff Behavior

**Files:**
- Modify: `src/elspeth/plugins/llm/aimd_throttle.py`
- Modify: `tests/plugins/llm/test_aimd_throttle.py`

Implement the multiplicative decrease on capacity errors.

### Step 1: Write failing test for backoff

```python
# Add to tests/plugins/llm/test_aimd_throttle.py

class TestAIMDThrottleBackoff:
    """Test multiplicative decrease on capacity errors."""

    def test_first_capacity_error_sets_initial_delay(self) -> None:
        """First error should set delay to initial backoff value."""
        throttle = AIMDThrottle()
        assert throttle.current_delay_ms == 0

        throttle.on_capacity_error()

        # First error with 0 delay should set to recovery_step (bootstrap)
        assert throttle.current_delay_ms == 50  # recovery_step default

    def test_subsequent_errors_multiply_delay(self) -> None:
        """Each error should multiply delay by backoff_multiplier."""
        config = ThrottleConfig(
            backoff_multiplier=2.0,
            recovery_step_ms=100,
        )
        throttle = AIMDThrottle(config)

        throttle.on_capacity_error()  # 0 -> 100
        assert throttle.current_delay_ms == 100

        throttle.on_capacity_error()  # 100 * 2 = 200
        assert throttle.current_delay_ms == 200

        throttle.on_capacity_error()  # 200 * 2 = 400
        assert throttle.current_delay_ms == 400

    def test_delay_capped_at_max(self) -> None:
        """Delay should not exceed max_dispatch_delay_ms."""
        config = ThrottleConfig(
            max_dispatch_delay_ms=500,
            backoff_multiplier=2.0,
            recovery_step_ms=100,
        )
        throttle = AIMDThrottle(config)

        # Drive delay up to cap
        for _ in range(10):
            throttle.on_capacity_error()

        assert throttle.current_delay_ms == 500
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleBackoff -v`
Expected: FAIL with "AttributeError: 'AIMDThrottle' object has no attribute 'on_capacity_error'"

### Step 3: Implement on_capacity_error

```python
# Add to AIMDThrottle class in src/elspeth/plugins/llm/aimd_throttle.py

    def on_capacity_error(self) -> None:
        """Record capacity error - multiply delay (thread-safe).

        If current delay is 0, bootstraps to recovery_step_ms.
        Otherwise multiplies by backoff_multiplier, capped at max.
        """
        with self._lock:
            if self._current_delay_ms == 0:
                # Bootstrap: start with recovery_step as initial backoff
                self._current_delay_ms = float(self._config.recovery_step_ms)
            else:
                # Multiplicative decrease
                self._current_delay_ms *= self._config.backoff_multiplier

            # Cap at maximum
            if self._current_delay_ms > self._config.max_dispatch_delay_ms:
                self._current_delay_ms = float(self._config.max_dispatch_delay_ms)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleBackoff -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/aimd_throttle.py tests/plugins/llm/test_aimd_throttle.py
git commit -m "feat(llm): add AIMD throttle backoff on capacity errors"
```

---

## Task 3: AIMD Throttle Recovery Behavior

**Files:**
- Modify: `src/elspeth/plugins/llm/aimd_throttle.py`
- Modify: `tests/plugins/llm/test_aimd_throttle.py`

Implement the additive increase (slow recovery) on success.

### Step 1: Write failing test for recovery

```python
# Add to tests/plugins/llm/test_aimd_throttle.py

class TestAIMDThrottleRecovery:
    """Test additive increase on success (slow recovery)."""

    def test_success_subtracts_recovery_step(self) -> None:
        """Each success should subtract recovery_step_ms."""
        config = ThrottleConfig(recovery_step_ms=50)
        throttle = AIMDThrottle(config)

        # Set initial delay
        throttle.on_capacity_error()  # -> 50
        throttle.on_capacity_error()  # -> 100
        assert throttle.current_delay_ms == 100

        throttle.on_success()  # 100 - 50 = 50
        assert throttle.current_delay_ms == 50

        throttle.on_success()  # 50 - 50 = 0
        assert throttle.current_delay_ms == 0

    def test_delay_floored_at_min(self) -> None:
        """Delay should not go below min_dispatch_delay_ms."""
        config = ThrottleConfig(
            min_dispatch_delay_ms=10,
            recovery_step_ms=100,
        )
        throttle = AIMDThrottle(config)

        # Set initial delay
        throttle.on_capacity_error()  # -> 100

        # Multiple successes should stop at min
        for _ in range(5):
            throttle.on_success()

        assert throttle.current_delay_ms == 10

    def test_success_at_zero_stays_zero(self) -> None:
        """Success when already at zero should stay at zero."""
        throttle = AIMDThrottle()
        assert throttle.current_delay_ms == 0

        throttle.on_success()

        assert throttle.current_delay_ms == 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleRecovery -v`
Expected: FAIL with "AttributeError: 'AIMDThrottle' object has no attribute 'on_success'"

### Step 3: Implement on_success

```python
# Add to AIMDThrottle class in src/elspeth/plugins/llm/aimd_throttle.py

    def on_success(self) -> None:
        """Record successful request - subtract recovery step (thread-safe).

        Subtracts recovery_step_ms from current delay, floored at min.
        """
        with self._lock:
            self._current_delay_ms -= self._config.recovery_step_ms

            # Floor at minimum
            if self._current_delay_ms < self._config.min_dispatch_delay_ms:
                self._current_delay_ms = float(self._config.min_dispatch_delay_ms)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleRecovery -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/aimd_throttle.py tests/plugins/llm/test_aimd_throttle.py
git commit -m "feat(llm): add AIMD throttle recovery on success"
```

---

## Task 4: AIMD Throttle Statistics

**Files:**
- Modify: `src/elspeth/plugins/llm/aimd_throttle.py`
- Modify: `tests/plugins/llm/test_aimd_throttle.py`

Add statistics tracking for audit trail.

### Step 1: Write failing test for stats

```python
# Add to tests/plugins/llm/test_aimd_throttle.py

class TestAIMDThrottleStats:
    """Test statistics tracking for audit."""

    def test_stats_track_capacity_retries(self) -> None:
        """Stats should count capacity retries."""
        throttle = AIMDThrottle()

        throttle.on_capacity_error()
        throttle.on_capacity_error()
        throttle.on_success()
        throttle.on_capacity_error()

        stats = throttle.get_stats()
        assert stats["capacity_retries"] == 3
        assert stats["successes"] == 1

    def test_stats_track_peak_delay(self) -> None:
        """Stats should track peak delay reached."""
        config = ThrottleConfig(
            max_dispatch_delay_ms=1000,
            backoff_multiplier=2.0,
            recovery_step_ms=50,
        )
        throttle = AIMDThrottle(config)

        throttle.on_capacity_error()  # 50
        throttle.on_capacity_error()  # 100
        throttle.on_capacity_error()  # 200
        throttle.on_success()         # 150
        throttle.on_success()         # 100

        stats = throttle.get_stats()
        assert stats["peak_delay_ms"] == 200
        assert stats["current_delay_ms"] == 100

    def test_stats_reset(self) -> None:
        """Stats can be reset."""
        throttle = AIMDThrottle()

        throttle.on_capacity_error()
        throttle.on_success()

        throttle.reset_stats()

        stats = throttle.get_stats()
        assert stats["capacity_retries"] == 0
        assert stats["successes"] == 0
        # current_delay is NOT reset - only counters
        assert stats["current_delay_ms"] == 0  # Was recovered to 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleStats -v`
Expected: FAIL with "AttributeError: 'AIMDThrottle' object has no attribute 'get_stats'"

### Step 3: Add stats tracking

```python
# Modify AIMDThrottle.__init__ and add methods in src/elspeth/plugins/llm/aimd_throttle.py

# Update __init__ to add:
        self._capacity_retries = 0
        self._successes = 0
        self._peak_delay_ms: float = 0.0

# Update on_capacity_error to track stats (add inside the lock):
            self._capacity_retries += 1
            if self._current_delay_ms > self._peak_delay_ms:
                self._peak_delay_ms = self._current_delay_ms

# Update on_success to track stats (add inside the lock):
            self._successes += 1

# Add new methods:
    def get_stats(self) -> dict[str, float | int]:
        """Get throttle statistics for audit trail (thread-safe).

        Returns:
            Dict with capacity_retries, successes, peak_delay_ms, current_delay_ms
        """
        with self._lock:
            return {
                "capacity_retries": self._capacity_retries,
                "successes": self._successes,
                "peak_delay_ms": self._peak_delay_ms,
                "current_delay_ms": self._current_delay_ms,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters (thread-safe).

        Note: Does NOT reset current_delay - only resets counters.
        """
        with self._lock:
            self._capacity_retries = 0
            self._successes = 0
            self._peak_delay_ms = self._current_delay_ms
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_aimd_throttle.py::TestAIMDThrottleStats -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/aimd_throttle.py tests/plugins/llm/test_aimd_throttle.py
git commit -m "feat(llm): add AIMD throttle statistics for audit"
```

---

## Task 5: Capacity Error Classification

**Files:**
- Create: `src/elspeth/plugins/llm/capacity_errors.py`
- Test: `tests/plugins/llm/test_capacity_errors.py`

Define which HTTP status codes are capacity errors vs normal errors.

### Step 1: Write failing test for classification

```python
# tests/plugins/llm/test_capacity_errors.py
"""Tests for capacity error classification."""

import pytest

from elspeth.plugins.llm.capacity_errors import (
    CAPACITY_ERROR_CODES,
    is_capacity_error,
    CapacityError,
)


class TestCapacityErrorClassification:
    """Test HTTP status code classification."""

    def test_429_is_capacity_error(self) -> None:
        """429 Too Many Requests is a capacity error."""
        assert is_capacity_error(429)
        assert 429 in CAPACITY_ERROR_CODES

    def test_503_is_capacity_error(self) -> None:
        """503 Service Unavailable is a capacity error."""
        assert is_capacity_error(503)
        assert 503 in CAPACITY_ERROR_CODES

    def test_529_is_capacity_error(self) -> None:
        """529 (Azure overloaded) is a capacity error."""
        assert is_capacity_error(529)
        assert 529 in CAPACITY_ERROR_CODES

    def test_500_is_not_capacity_error(self) -> None:
        """500 Internal Server Error is NOT a capacity error."""
        assert not is_capacity_error(500)
        assert 500 not in CAPACITY_ERROR_CODES

    def test_400_is_not_capacity_error(self) -> None:
        """400 Bad Request is NOT a capacity error."""
        assert not is_capacity_error(400)

    def test_401_is_not_capacity_error(self) -> None:
        """401 Unauthorized is NOT a capacity error."""
        assert not is_capacity_error(401)

    def test_200_is_not_capacity_error(self) -> None:
        """200 OK is NOT a capacity error."""
        assert not is_capacity_error(200)


class TestCapacityErrorException:
    """Test CapacityError exception."""

    def test_capacity_error_stores_status_code(self) -> None:
        """CapacityError should store the status code."""
        error = CapacityError(429, "Rate limited")

        assert error.status_code == 429
        assert str(error) == "Rate limited"

    def test_capacity_error_retryable_flag(self) -> None:
        """CapacityError should always be retryable."""
        error = CapacityError(503, "Service unavailable")

        assert error.retryable is True
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_capacity_errors.py -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement capacity error classification

```python
# src/elspeth/plugins/llm/capacity_errors.py
"""Capacity error classification for LLM API calls.

Capacity errors are transient overload conditions that should be retried
indefinitely with AIMD throttling. They are distinct from "normal" errors
(auth failures, malformed requests) which use standard retry limits.

HTTP Status Codes:
- 429: Too Many Requests (universal)
- 503: Service Unavailable (universal)
- 529: Overloaded (Azure, some other providers)
"""

from __future__ import annotations


# HTTP status codes that indicate capacity/rate limiting
# These trigger AIMD throttle and infinite retry
CAPACITY_ERROR_CODES: frozenset[int] = frozenset({429, 503, 529})


def is_capacity_error(status_code: int) -> bool:
    """Check if HTTP status code indicates a capacity error.

    Capacity errors are transient overload conditions that should trigger
    AIMD throttle backoff and be retried indefinitely.

    Args:
        status_code: HTTP status code

    Returns:
        True if this is a capacity error, False otherwise
    """
    return status_code in CAPACITY_ERROR_CODES


class CapacityError(Exception):
    """Exception for capacity/rate limit errors.

    Raised when an LLM API call fails due to capacity limits.
    These errors should NEVER cause row failure - they are always retried.

    Attributes:
        status_code: HTTP status code that triggered this error
        retryable: Always True for capacity errors
    """

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize capacity error.

        Args:
            status_code: HTTP status code (429, 503, or 529)
            message: Error message
        """
        super().__init__(message)
        self.status_code = status_code
        self.retryable = True
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_capacity_errors.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/capacity_errors.py tests/plugins/llm/test_capacity_errors.py
git commit -m "feat(llm): add capacity error classification"
```

---

## Task 6: Reorder Buffer

**Files:**
- Create: `src/elspeth/plugins/llm/reorder_buffer.py`
- Test: `tests/plugins/llm/test_reorder_buffer.py`

Buffer that maintains strict submission order regardless of completion order.

### Step 1: Write failing test for buffer

```python
# tests/plugins/llm/test_reorder_buffer.py
"""Tests for reorder buffer that maintains submission order."""

import pytest
from concurrent.futures import Future
from unittest.mock import MagicMock

from elspeth.plugins.llm.reorder_buffer import ReorderBuffer


class TestReorderBufferBasic:
    """Test basic reorder buffer operations."""

    def test_empty_buffer_has_no_ready_results(self) -> None:
        """Empty buffer should have no ready results."""
        buffer = ReorderBuffer[str]()

        assert buffer.get_ready_results() == []
        assert buffer.pending_count == 0

    def test_single_result_emitted_immediately(self) -> None:
        """Single result should be available immediately."""
        buffer = ReorderBuffer[str]()

        idx = buffer.submit()
        buffer.complete(idx, "result_0")

        results = buffer.get_ready_results()
        assert results == ["result_0"]
        assert buffer.pending_count == 0


class TestReorderBufferOrdering:
    """Test that results are emitted in submission order."""

    def test_out_of_order_completion_reordered(self) -> None:
        """Results completing out of order should be emitted in order."""
        buffer = ReorderBuffer[str]()

        # Submit 5 items
        indices = [buffer.submit() for _ in range(5)]
        assert indices == [0, 1, 2, 3, 4]

        # Complete in order: 2, 0, 4, 1, 3
        buffer.complete(2, "result_2")
        assert buffer.get_ready_results() == []  # Can't emit yet

        buffer.complete(0, "result_0")
        assert buffer.get_ready_results() == ["result_0"]  # 0 ready, 1 blocks 2

        buffer.complete(4, "result_4")
        assert buffer.get_ready_results() == []  # Still waiting for 1

        buffer.complete(1, "result_1")
        # Now 1 and 2 can be emitted
        assert buffer.get_ready_results() == ["result_1", "result_2"]

        buffer.complete(3, "result_3")
        # Now 3 and 4 can be emitted
        assert buffer.get_ready_results() == ["result_3", "result_4"]

        assert buffer.pending_count == 0

    def test_in_order_completion_immediate(self) -> None:
        """Results completing in order should emit immediately."""
        buffer = ReorderBuffer[str]()

        for i in range(3):
            idx = buffer.submit()
            buffer.complete(idx, f"result_{i}")
            assert buffer.get_ready_results() == [f"result_{i}"]

        assert buffer.pending_count == 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_reorder_buffer.py -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement reorder buffer

```python
# src/elspeth/plugins/llm/reorder_buffer.py
"""Reorder buffer for maintaining strict submission order.

Results may complete out of order (due to varying API latencies),
but are emitted in the exact order they were submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _BufferEntry(Generic[T]):
    """Entry in the reorder buffer."""

    submit_index: int
    complete_index: int | None = None
    result: T | None = None
    is_complete: bool = False


class ReorderBuffer(Generic[T]):
    """Thread-safe buffer that reorders results to match submission order.

    Usage:
        buffer = ReorderBuffer[TransformResult]()

        # Submit work (returns index)
        idx = buffer.submit()

        # ... do async work ...

        # Complete with result (may be out of order)
        buffer.complete(idx, result)

        # Get results in submission order
        ready = buffer.get_ready_results()
    """

    def __init__(self) -> None:
        """Initialize empty buffer."""
        self._entries: dict[int, _BufferEntry[T]] = {}
        self._next_submit: int = 0
        self._next_emit: int = 0
        self._complete_counter: int = 0
        self._lock = Lock()

    @property
    def pending_count(self) -> int:
        """Number of submitted but not-yet-emitted items (thread-safe)."""
        with self._lock:
            return self._next_submit - self._next_emit

    def submit(self) -> int:
        """Reserve a slot and return its index (thread-safe).

        Returns:
            Index to use when completing this item
        """
        with self._lock:
            idx = self._next_submit
            self._entries[idx] = _BufferEntry(submit_index=idx)
            self._next_submit += 1
            return idx

    def complete(self, index: int, result: T) -> None:
        """Mark an item as complete with its result (thread-safe).

        Args:
            index: Index returned from submit()
            result: The result for this item

        Raises:
            KeyError: If index was never submitted
            ValueError: If index was already completed
        """
        with self._lock:
            if index not in self._entries:
                raise KeyError(f"Index {index} was never submitted")

            entry = self._entries[index]
            if entry.is_complete:
                raise ValueError(f"Index {index} was already completed")

            entry.result = result
            entry.complete_index = self._complete_counter
            entry.is_complete = True
            self._complete_counter += 1

    def get_ready_results(self) -> list[T]:
        """Get all results that are ready to emit in order (thread-safe).

        Returns results that are:
        1. Complete (result received)
        2. All previous indices are also complete

        Returns:
            List of results in submission order (may be empty)
        """
        with self._lock:
            ready: list[T] = []

            while self._next_emit in self._entries:
                entry = self._entries[self._next_emit]
                if not entry.is_complete:
                    break

                # Entry is complete and all previous are emitted
                ready.append(entry.result)  # type: ignore[arg-type]
                del self._entries[self._next_emit]
                self._next_emit += 1

            return ready
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_reorder_buffer.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/reorder_buffer.py tests/plugins/llm/test_reorder_buffer.py
git commit -m "feat(llm): add reorder buffer for strict output ordering"
```

---

## Task 7: Pool Configuration Schema

**Files:**
- Modify: `src/elspeth/plugins/llm/base.py`
- Test: `tests/plugins/llm/test_pool_config.py`

Add pool configuration fields to LLMConfig.

### Step 1: Write failing test for config

```python
# tests/plugins/llm/test_pool_config.py
"""Tests for pool configuration in LLM transforms."""

import pytest

from elspeth.plugins.llm.base import LLMConfig, PoolConfig


class TestPoolConfigDefaults:
    """Test pool configuration defaults."""

    def test_default_pool_size_is_sequential(self) -> None:
        """Default pool_size=1 means sequential processing."""
        config = LLMConfig.from_dict({
            "model": "gpt-4",
            "template": "{{ row.text }}",
            "schema": {"fields": "dynamic"},
        })

        assert config.pool_config is None or config.pool_config.pool_size == 1

    def test_pool_size_1_is_sequential_mode(self) -> None:
        """pool_size=1 should not create pool config."""
        config = LLMConfig.from_dict({
            "model": "gpt-4",
            "template": "{{ row.text }}",
            "schema": {"fields": "dynamic"},
            "pool_size": 1,
        })

        # pool_size=1 means sequential, no pooling needed
        assert config.pool_config is None


class TestPoolConfigExplicit:
    """Test explicit pool configuration."""

    def test_pool_size_greater_than_1_creates_config(self) -> None:
        """pool_size > 1 should create pool config with defaults."""
        config = LLMConfig.from_dict({
            "model": "gpt-4",
            "template": "{{ row.text }}",
            "schema": {"fields": "dynamic"},
            "pool_size": 10,
        })

        assert config.pool_config is not None
        assert config.pool_config.pool_size == 10
        # AIMD defaults
        assert config.pool_config.min_dispatch_delay_ms == 0
        assert config.pool_config.max_dispatch_delay_ms == 5000
        assert config.pool_config.backoff_multiplier == 2.0
        assert config.pool_config.recovery_step_ms == 50

    def test_custom_aimd_settings(self) -> None:
        """Custom AIMD settings should be applied."""
        config = LLMConfig.from_dict({
            "model": "gpt-4",
            "template": "{{ row.text }}",
            "schema": {"fields": "dynamic"},
            "pool_size": 5,
            "min_dispatch_delay_ms": 10,
            "max_dispatch_delay_ms": 1000,
            "backoff_multiplier": 3.0,
            "recovery_step_ms": 25,
        })

        assert config.pool_config is not None
        assert config.pool_config.pool_size == 5
        assert config.pool_config.min_dispatch_delay_ms == 10
        assert config.pool_config.max_dispatch_delay_ms == 1000
        assert config.pool_config.backoff_multiplier == 3.0
        assert config.pool_config.recovery_step_ms == 25


class TestPoolConfigValidation:
    """Test pool configuration validation."""

    def test_pool_size_must_be_positive(self) -> None:
        """pool_size must be >= 1."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig.from_dict({
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": {"fields": "dynamic"},
                "pool_size": 0,
            })

    def test_backoff_multiplier_must_be_greater_than_1(self) -> None:
        """backoff_multiplier must be > 1."""
        with pytest.raises(Exception):
            LLMConfig.from_dict({
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": {"fields": "dynamic"},
                "pool_size": 10,
                "backoff_multiplier": 0.5,
            })
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_pool_config.py -v`
Expected: FAIL with "ImportError" for PoolConfig

### Step 3: Add pool config to LLMConfig

```python
# Add to src/elspeth/plugins/llm/base.py

# Add new import at top:
from elspeth.plugins.llm.aimd_throttle import ThrottleConfig

# Add PoolConfig class before LLMConfig:
class PoolConfig(BaseModel):
    """Configuration for parallel request pooling.

    When pool_size > 1, requests are dispatched in parallel with
    AIMD throttling for adaptive rate control.
    """

    model_config = {"extra": "forbid"}

    pool_size: int = Field(
        1, ge=1, description="Max concurrent requests (1 = sequential)"
    )
    min_dispatch_delay_ms: int = Field(
        0, ge=0, description="Floor for delay between dispatches"
    )
    max_dispatch_delay_ms: int = Field(
        5000, ge=0, description="Ceiling for delay"
    )
    backoff_multiplier: float = Field(
        2.0, gt=1.0, description="Multiply delay on capacity error"
    )
    recovery_step_ms: int = Field(
        50, ge=0, description="Subtract from delay on success"
    )

    def to_throttle_config(self) -> ThrottleConfig:
        """Convert to ThrottleConfig for AIMD throttle."""
        return ThrottleConfig(
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
        )


# Add to LLMConfig class (new fields):
    # Pool configuration (optional - extracted from flat fields)
    pool_size: int = Field(1, ge=1, description="Max concurrent requests")
    min_dispatch_delay_ms: int = Field(0, ge=0)
    max_dispatch_delay_ms: int = Field(5000, ge=0)
    backoff_multiplier: float = Field(2.0, gt=1.0)
    recovery_step_ms: int = Field(50, ge=0)

    @property
    def pool_config(self) -> PoolConfig | None:
        """Get pool configuration if pooling is enabled.

        Returns None if pool_size=1 (sequential mode).
        """
        if self.pool_size <= 1:
            return None
        return PoolConfig(
            pool_size=self.pool_size,
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
        )
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_pool_config.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/base.py tests/plugins/llm/test_pool_config.py
git commit -m "feat(llm): add pool configuration to LLMConfig"
```

---

## Task 8: PooledExecutor Core Structure

**Files:**
- Create: `src/elspeth/plugins/llm/pooled_executor.py`
- Test: `tests/plugins/llm/test_pooled_executor.py`

Create the main executor class structure.

### Step 1: Write failing test for executor

```python
# tests/plugins/llm/test_pooled_executor.py
"""Tests for PooledExecutor parallel request handling."""

import pytest
from unittest.mock import MagicMock

from elspeth.contracts import TransformResult
from elspeth.plugins.llm.pooled_executor import PooledExecutor
from elspeth.plugins.llm.base import PoolConfig


class TestPooledExecutorInit:
    """Test executor initialization."""

    def test_creates_with_config(self) -> None:
        """Executor should accept pool config."""
        config = PoolConfig(pool_size=10)

        executor = PooledExecutor(config)

        assert executor.pool_size == 10
        assert executor.pending_count == 0

    def test_creates_throttle_from_config(self) -> None:
        """Executor should create AIMD throttle from config."""
        config = PoolConfig(
            pool_size=5,
            backoff_multiplier=3.0,
            recovery_step_ms=100,
        )

        executor = PooledExecutor(config)

        assert executor._throttle.config.backoff_multiplier == 3.0
        assert executor._throttle.config.recovery_step_ms == 100


class TestPooledExecutorShutdown:
    """Test executor shutdown."""

    def test_shutdown_completes_pending(self) -> None:
        """Shutdown should wait for pending requests."""
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        # Should not raise
        executor.shutdown(wait=True)

        assert executor.pending_count == 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorInit -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement executor structure

```python
# src/elspeth/plugins/llm/pooled_executor.py
"""Pooled executor for parallel LLM API calls with AIMD throttling.

Manages concurrent requests while:
- Respecting pool size limits via semaphore
- Applying AIMD throttle delays between dispatches
- Reordering results to match submission order
- Tracking statistics for audit trail
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
from typing import TYPE_CHECKING, Any, Callable

from elspeth.contracts import TransformResult
from elspeth.plugins.llm.aimd_throttle import AIMDThrottle
from elspeth.plugins.llm.base import PoolConfig
from elspeth.plugins.llm.reorder_buffer import ReorderBuffer

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext


class PooledExecutor:
    """Executor for parallel LLM API calls with strict ordering.

    Manages a pool of concurrent requests with:
    - Semaphore-controlled dispatch (max pool_size in flight)
    - AIMD throttle for adaptive rate limiting
    - Reorder buffer for strict submission order output

    The executor is synchronous from the caller's perspective -
    execute_batch() blocks until all results are ready in order.

    Usage:
        executor = PooledExecutor(pool_config)

        # Process batch of rows
        results = executor.execute_batch(
            rows=[row1, row2, ...],
            process_fn=lambda row, ctx: transform.process_single(row, ctx),
            ctx=plugin_context,
        )

        # Results are in submission order
        assert len(results) == len(rows)

        # Get stats for audit
        stats = executor.get_stats()
    """

    def __init__(self, config: PoolConfig) -> None:
        """Initialize executor with pool configuration.

        Args:
            config: Pool configuration with size and AIMD settings
        """
        self._config = config
        self._pool_size = config.pool_size

        # Thread pool for concurrent execution
        self._thread_pool = ThreadPoolExecutor(max_workers=config.pool_size)

        # Semaphore limits concurrent in-flight requests
        self._semaphore = Semaphore(config.pool_size)

        # AIMD throttle for adaptive rate control
        self._throttle = AIMDThrottle(config.to_throttle_config())

        # Reorder buffer for strict output ordering
        self._buffer: ReorderBuffer[TransformResult] = ReorderBuffer()

        self._shutdown = False

    @property
    def pool_size(self) -> int:
        """Maximum concurrent requests."""
        return self._pool_size

    @property
    def pending_count(self) -> int:
        """Number of requests in flight or buffered."""
        return self._buffer.pending_count

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for pending requests to complete
        """
        self._shutdown = True
        self._thread_pool.shutdown(wait=wait)

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics for audit trail.

        Returns:
            Dict with pool_size, throttle stats, etc.
        """
        throttle_stats = self._throttle.get_stats()
        return {
            "pool_config": {
                "pool_size": self._pool_size,
            },
            "pool_stats": {
                "capacity_retries": throttle_stats["capacity_retries"],
                "successes": throttle_stats["successes"],
                "peak_delay_ms": throttle_stats["peak_delay_ms"],
                "current_delay_ms": throttle_stats["current_delay_ms"],
            },
        }
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_pooled_executor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/pooled_executor.py tests/plugins/llm/test_pooled_executor.py
git commit -m "feat(llm): add PooledExecutor core structure"
```

---

## Task 9: PooledExecutor Batch Execution

**Files:**
- Modify: `src/elspeth/plugins/llm/pooled_executor.py`
- Modify: `tests/plugins/llm/test_pooled_executor.py`

Implement the main execute_batch method.

### Step 1: Write failing test for batch execution

```python
# Add to tests/plugins/llm/test_pooled_executor.py

import time
from concurrent.futures import Future

class TestPooledExecutorBatch:
    """Test batch execution with ordering."""

    def test_execute_batch_returns_results_in_order(self) -> None:
        """Results should be in submission order regardless of completion."""
        config = PoolConfig(pool_size=3)
        executor = PooledExecutor(config)

        # Mock process function with varying delays
        call_order: list[int] = []

        def mock_process(row: dict, ctx: Any) -> TransformResult:
            idx = row["idx"]
            call_order.append(idx)
            # Varying delays to cause out-of-order completion
            time.sleep(0.01 * (3 - idx))  # idx 0 slowest, idx 2 fastest
            return TransformResult.success({"idx": idx, "result": f"done_{idx}"})

        rows = [{"idx": i} for i in range(3)]
        ctx = MagicMock()

        results = executor.execute_batch(rows, mock_process, ctx)

        # Results must be in submission order
        assert len(results) == 3
        assert results[0].row["idx"] == 0
        assert results[1].row["idx"] == 1
        assert results[2].row["idx"] == 2

        executor.shutdown()

    def test_execute_batch_respects_pool_size(self) -> None:
        """Should never exceed pool_size concurrent requests."""
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        max_concurrent = 0
        current_concurrent = 0
        lock = __import__("threading").Lock()

        def mock_process(row: dict, ctx: Any) -> TransformResult:
            nonlocal max_concurrent, current_concurrent
            with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent

            time.sleep(0.05)

            with lock:
                current_concurrent -= 1

            return TransformResult.success(row)

        rows = [{"idx": i} for i in range(5)]
        ctx = MagicMock()

        results = executor.execute_batch(rows, mock_process, ctx)

        assert len(results) == 5
        assert max_concurrent <= 2  # Never exceeded pool_size

        executor.shutdown()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorBatch -v`
Expected: FAIL with "AttributeError: 'PooledExecutor' object has no attribute 'execute_batch'"

### Step 3: Implement execute_batch

```python
# Add to PooledExecutor class in src/elspeth/plugins/llm/pooled_executor.py

import time
from concurrent.futures import Future, as_completed

    def execute_batch(
        self,
        rows: list[dict[str, Any]],
        process_fn: Callable[[dict[str, Any], PluginContext], TransformResult],
        ctx: PluginContext,
    ) -> list[TransformResult]:
        """Execute batch of rows with parallel processing.

        Dispatches rows to the thread pool with semaphore control,
        applies AIMD throttle delays, and returns results in
        submission order.

        Args:
            rows: List of row dicts to process
            process_fn: Function that processes a single row
            ctx: Plugin context for the process function

        Returns:
            List of TransformResults in same order as input rows
        """
        if not rows:
            return []

        # Track futures by their buffer index
        futures: dict[Future[tuple[int, TransformResult]], int] = {}

        # Submit all rows
        for row in rows:
            # Reserve slot in reorder buffer
            buffer_idx = self._buffer.submit()

            # Apply throttle delay before dispatch
            delay_ms = self._throttle.current_delay_ms
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)

            # Acquire semaphore (blocks if pool is full)
            self._semaphore.acquire()

            # Submit to thread pool
            future = self._thread_pool.submit(
                self._execute_single,
                buffer_idx,
                row,
                process_fn,
                ctx,
            )
            futures[future] = buffer_idx

        # Wait for all futures and collect results
        results: list[TransformResult] = []

        for future in as_completed(futures):
            buffer_idx, result = future.result()

            # Complete in buffer (may be out of order)
            self._buffer.complete(buffer_idx, result)

            # Collect any ready results
            ready = self._buffer.get_ready_results()
            results.extend(ready)

        return results

    def _execute_single(
        self,
        buffer_idx: int,
        row: dict[str, Any],
        process_fn: Callable[[dict[str, Any], PluginContext], TransformResult],
        ctx: PluginContext,
    ) -> tuple[int, TransformResult]:
        """Execute single row and handle throttle feedback.

        Args:
            buffer_idx: Index in reorder buffer
            row: Row to process
            process_fn: Processing function
            ctx: Plugin context

        Returns:
            Tuple of (buffer_idx, result)
        """
        try:
            result = process_fn(row, ctx)
            self._throttle.on_success()
            return (buffer_idx, result)
        finally:
            # Always release semaphore
            self._semaphore.release()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorBatch -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/pooled_executor.py tests/plugins/llm/test_pooled_executor.py
git commit -m "feat(llm): add PooledExecutor batch execution"
```

---

## Task 10: PooledExecutor Capacity Error Handling

**Files:**
- Modify: `src/elspeth/plugins/llm/pooled_executor.py`
- Modify: `tests/plugins/llm/test_pooled_executor.py`

Add capacity error detection and infinite retry with throttle.

### Step 1: Write failing test for capacity handling

```python
# Add to tests/plugins/llm/test_pooled_executor.py

from elspeth.plugins.llm.capacity_errors import CapacityError

class TestPooledExecutorCapacityHandling:
    """Test capacity error handling with AIMD throttle."""

    def test_capacity_error_triggers_throttle_and_retries(self) -> None:
        """Capacity errors should trigger throttle and retry."""
        config = PoolConfig(pool_size=2, recovery_step_ms=50)
        executor = PooledExecutor(config)

        call_count = 0

        def mock_process(row: dict, ctx: Any) -> TransformResult:
            nonlocal call_count
            call_count += 1

            # First call raises capacity error, second succeeds
            if call_count == 1:
                raise CapacityError(429, "Rate limited")
            return TransformResult.success(row)

        rows = [{"idx": 0}]
        ctx = MagicMock()

        results = executor.execute_batch(rows, mock_process, ctx)

        # Should have retried and succeeded
        assert len(results) == 1
        assert results[0].status == "success"
        assert call_count == 2

        # Throttle should have been triggered
        stats = executor.get_stats()
        assert stats["pool_stats"]["capacity_retries"] == 1

        executor.shutdown()

    def test_capacity_errors_never_fail_row(self) -> None:
        """Rows should never fail due to capacity errors."""
        config = PoolConfig(pool_size=1, max_dispatch_delay_ms=100)
        executor = PooledExecutor(config)

        call_count = 0

        def mock_process(row: dict, ctx: Any) -> TransformResult:
            nonlocal call_count
            call_count += 1

            # Fail 3 times with capacity error, then succeed
            if call_count <= 3:
                raise CapacityError(503, "Service unavailable")
            return TransformResult.success(row)

        rows = [{"idx": 0}]
        ctx = MagicMock()

        results = executor.execute_batch(rows, mock_process, ctx)

        assert len(results) == 1
        assert results[0].status == "success"
        assert call_count == 4

        executor.shutdown()

    def test_normal_error_not_infinitely_retried(self) -> None:
        """Non-capacity errors should not be infinitely retried."""
        config = PoolConfig(pool_size=1)
        executor = PooledExecutor(config)

        def mock_process(row: dict, ctx: Any) -> TransformResult:
            # Return error result (not raise CapacityError)
            return TransformResult.error({"reason": "bad_request"})

        rows = [{"idx": 0}]
        ctx = MagicMock()

        results = executor.execute_batch(rows, mock_process, ctx)

        # Should return error without infinite retry
        assert len(results) == 1
        assert results[0].status == "error"

        executor.shutdown()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorCapacityHandling -v`
Expected: FAIL - capacity errors not handled

### Step 3: Add capacity error handling

```python
# Modify _execute_single in src/elspeth/plugins/llm/pooled_executor.py

# Add import at top:
from elspeth.plugins.llm.capacity_errors import CapacityError

    def _execute_single(
        self,
        buffer_idx: int,
        row: dict[str, Any],
        process_fn: Callable[[dict[str, Any], PluginContext], TransformResult],
        ctx: PluginContext,
    ) -> tuple[int, TransformResult]:
        """Execute single row with capacity error retry.

        Capacity errors trigger AIMD throttle and infinite retry.
        Normal errors/results are returned as-is.
        """
        try:
            while True:
                try:
                    result = process_fn(row, ctx)
                    self._throttle.on_success()
                    return (buffer_idx, result)
                except CapacityError:
                    # Trigger throttle backoff
                    self._throttle.on_capacity_error()

                    # Wait throttle delay before retry
                    delay_ms = self._throttle.current_delay_ms
                    if delay_ms > 0:
                        time.sleep(delay_ms / 1000)

                    # Retry (infinite for capacity errors)
                    continue
        finally:
            # Always release semaphore
            self._semaphore.release()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorCapacityHandling -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/pooled_executor.py tests/plugins/llm/test_pooled_executor.py
git commit -m "feat(llm): add capacity error handling with infinite retry"
```

---

## Task 11: Integrate PooledExecutor into OpenRouterLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter.py`
- Modify: `tests/integration/test_llm_transforms.py`

Wire up the pooled executor to the existing transform.

### Step 1: Write failing integration test

```python
# Add to tests/integration/test_llm_transforms.py

class TestOpenRouterPooledExecution:
    """Integration tests for OpenRouter with pooled execution."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    def test_pool_size_1_uses_sequential_processing(
        self, recorder: LandscapeRecorder
    ) -> None:
        """pool_size=1 should use existing sequential logic."""
        from unittest.mock import patch

        # Setup state
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="openrouter_llm",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"text": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"text": "test"},
        )

        transform = OpenRouterLLMTransform({
            "model": "anthropic/claude-3-opus",
            "template": "{{ row.text }}",
            "api_key": "test-key",
            "schema": {"fields": "dynamic"},
            "pool_size": 1,  # Sequential
        })

        # Verify no executor created
        assert transform._executor is None

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            landscape=recorder,
            state_id=state.state_id,
        )

        # Mock HTTP and verify single-row processing still works
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "result"}}],
            "usage": {},
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b""
        mock_response.text = ""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"

    def test_pool_size_greater_than_1_creates_executor(self) -> None:
        """pool_size > 1 should create pooled executor."""
        transform = OpenRouterLLMTransform({
            "model": "anthropic/claude-3-opus",
            "template": "{{ row.text }}",
            "api_key": "test-key",
            "schema": {"fields": "dynamic"},
            "pool_size": 5,
        })

        assert transform._executor is not None
        assert transform._executor.pool_size == 5
```

### Step 2: Run test to verify it fails

Run: `pytest tests/integration/test_llm_transforms.py::TestOpenRouterPooledExecution -v`
Expected: FAIL - _executor attribute doesn't exist

### Step 3: Integrate executor into OpenRouterLLMTransform

```python
# Modify src/elspeth/plugins/llm/openrouter.py

# Add imports at top:
from elspeth.plugins.llm.pooled_executor import PooledExecutor
from elspeth.plugins.llm.capacity_errors import CapacityError, is_capacity_error

# Modify __init__ to create executor:
    def __init__(self, config: dict[str, Any]) -> None:
        # ... existing code ...

        # Create pooled executor if pool_size > 1
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

# Add method for single-row processing (extract from process):
    def _process_single(
        self, row: dict[str, Any], ctx: PluginContext
    ) -> TransformResult:
        """Process a single row via OpenRouter API.

        This is the core processing logic, used both for sequential
        processing and as the worker function for pooled execution.

        Raises:
            CapacityError: On 429/503/529 HTTP errors (for pooled retry)
        """
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error({
                "reason": "template_rendering_failed",
                "error": str(e),
                "template_hash": self._template.template_hash,
                "template_source": self._template.template_source,
            })

        # 2. Build request
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens:
            request_body["max_tokens"] = self._max_tokens

        # 3. Call via audited HTTP client (EXTERNAL - wrap)
        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError(
                "OpenRouter transform requires landscape recorder and state_id."
            )

        http_client = AuditedHTTPClient(
            recorder=ctx.landscape,
            state_id=ctx.state_id,
            timeout=self._timeout,
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

        try:
            response = http_client.post(
                "/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Check for capacity error
            if is_capacity_error(e.response.status_code):
                raise CapacityError(
                    e.response.status_code, str(e)
                ) from e
            # Non-capacity HTTP error
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=False,
            )
        except httpx.RequestError as e:
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=False,
            )

        # 4. Parse JSON response (EXTERNAL DATA - wrap)
        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            return TransformResult.error({
                "reason": "invalid_json_response",
                "error": f"Response is not valid JSON: {e}",
                "content_type": response.headers.get("content-type", "unknown"),
                "body_preview": response.text[:500] if response.text else None,
            }, retryable=False)

        # 5. Extract content
        try:
            choices = data["choices"]
            if not choices:
                return TransformResult.error(
                    {"reason": "empty_choices", "response": data},
                    retryable=False,
                )
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            return TransformResult.error({
                "reason": "malformed_response",
                "error": f"{type(e).__name__}: {e}",
                "response_keys": list(data.keys()) if isinstance(data, dict) else None,
            }, retryable=False)

        usage = data.get("usage", {})

        output = dict(row)
        output[self._response_field] = content
        output[f"{self._response_field}_usage"] = usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_model"] = data.get("model", self._model)

        return TransformResult.success(output)

# Modify process to use executor when available:
    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process row via OpenRouter API.

        Uses pooled executor if pool_size > 1, otherwise sequential.
        """
        # Sequential mode - use existing logic
        return self._process_single(row, ctx)

    def close(self) -> None:
        """Release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/integration/test_llm_transforms.py::TestOpenRouterPooledExecution -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/openrouter.py tests/integration/test_llm_transforms.py
git commit -m "feat(llm): integrate PooledExecutor into OpenRouterLLMTransform"
```

---

## Task 12: Property-Based Test for Reorder Buffer

**Files:**
- Modify: `tests/plugins/llm/test_reorder_buffer.py`

Add Hypothesis property test for ordering invariant.

### Step 1: Write property test

```python
# Add to tests/plugins/llm/test_reorder_buffer.py

from hypothesis import given, strategies as st, settings


class TestReorderBufferProperties:
    """Property-based tests for reorder buffer invariants."""

    @given(
        completion_order=st.permutations(range(10)),
    )
    @settings(max_examples=100)
    def test_output_order_matches_submission_order(
        self, completion_order: list[int]
    ) -> None:
        """For ANY completion order, output is always in submission order."""
        buffer = ReorderBuffer[int]()
        n = len(completion_order)

        # Submit n items
        indices = [buffer.submit() for _ in range(n)]

        # Complete in permuted order
        for complete_idx in completion_order:
            buffer.complete(complete_idx, complete_idx)

        # Collect all results
        all_results: list[int] = []
        while buffer.pending_count > 0 or (ready := buffer.get_ready_results()):
            all_results.extend(ready)

        # Must be in submission order (0, 1, 2, ..., n-1)
        assert all_results == list(range(n))

    @given(
        n=st.integers(min_value=1, max_value=50),
    )
    def test_all_submitted_items_eventually_emitted(self, n: int) -> None:
        """Every submitted item is eventually emitted exactly once."""
        import random

        buffer = ReorderBuffer[str]()

        # Submit n items
        for _ in range(n):
            buffer.submit()

        # Complete in random order
        indices = list(range(n))
        random.shuffle(indices)
        for idx in indices:
            buffer.complete(idx, f"result_{idx}")

        # Collect all results
        all_results: list[str] = []
        while True:
            ready = buffer.get_ready_results()
            if not ready:
                break
            all_results.extend(ready)

        # Must have exactly n results
        assert len(all_results) == n
        assert buffer.pending_count == 0
```

### Step 2: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_reorder_buffer.py::TestReorderBufferProperties -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/plugins/llm/test_reorder_buffer.py
git commit -m "test(llm): add property-based tests for reorder buffer"
```

---

## Task 13: Update Module Exports

**Files:**
- Modify: `src/elspeth/plugins/llm/__init__.py`

Export new classes from the module.

### Step 1: Read current exports

Run: `cat src/elspeth/plugins/llm/__init__.py`

### Step 2: Add new exports

```python
# src/elspeth/plugins/llm/__init__.py
"""LLM transform plugins for ELSPETH."""

from elspeth.plugins.llm.aimd_throttle import AIMDThrottle, ThrottleConfig
from elspeth.plugins.llm.azure import AzureOpenAILLMTransform
from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform
from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig, PoolConfig
from elspeth.plugins.llm.capacity_errors import (
    CAPACITY_ERROR_CODES,
    CapacityError,
    is_capacity_error,
)
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform
from elspeth.plugins.llm.pooled_executor import PooledExecutor
from elspeth.plugins.llm.reorder_buffer import ReorderBuffer
from elspeth.plugins.llm.templates import PromptTemplate, RenderedPrompt, TemplateError

__all__ = [
    # Transforms
    "AzureBatchLLMTransform",
    "AzureOpenAILLMTransform",
    "BaseLLMTransform",
    "OpenRouterLLMTransform",
    # Config
    "LLMConfig",
    "PoolConfig",
    "ThrottleConfig",
    # Pooled execution
    "AIMDThrottle",
    "CapacityError",
    "CAPACITY_ERROR_CODES",
    "is_capacity_error",
    "PooledExecutor",
    "ReorderBuffer",
    # Templates
    "PromptTemplate",
    "RenderedPrompt",
    "TemplateError",
]
```

### Step 3: Run full test suite

Run: `pytest tests/plugins/llm/ -v`
Expected: All tests PASS

### Step 4: Commit

```bash
git add src/elspeth/plugins/llm/__init__.py
git commit -m "feat(llm): export pooled execution classes"
```

---

## Task 14: Full Integration Test with Throttle

**Files:**
- Modify: `tests/integration/test_llm_transforms.py`

Add integration test for full pooled execution with simulated capacity errors.

### Step 1: Write integration test

```python
# Add to tests/integration/test_llm_transforms.py

class TestPooledExecutionIntegration:
    """Full integration tests for pooled LLM execution."""

    def test_batch_with_simulated_capacity_errors(self) -> None:
        """Verify pooled execution handles capacity errors correctly."""
        from unittest.mock import patch, MagicMock
        import random

        # Create transform with pooling
        transform = OpenRouterLLMTransform({
            "model": "test-model",
            "template": "{{ row.text }}",
            "api_key": "test-key",
            "schema": {"fields": "dynamic"},
            "pool_size": 3,
            "max_dispatch_delay_ms": 100,
        })

        # Track calls per row
        call_counts: dict[int, int] = {}

        def mock_post(*args, **kwargs):
            """Simulate 30% capacity error rate."""
            body = kwargs.get("json", {})
            messages = body.get("messages", [])

            # Extract row index from message
            if messages:
                content = messages[-1].get("content", "")
                idx = int(content.split("_")[1]) if "_" in content else 0
                call_counts[idx] = call_counts.get(idx, 0) + 1

                # First call has 50% chance of capacity error
                if call_counts[idx] == 1 and random.random() < 0.5:
                    response = MagicMock(spec=httpx.Response)
                    response.status_code = 429
                    raise httpx.HTTPStatusError(
                        "Rate limited",
                        request=MagicMock(),
                        response=response,
                    )

            # Success response
            response = MagicMock(spec=httpx.Response)
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {
                "choices": [{"message": {"content": f"done"}}],
                "usage": {},
            }
            response.raise_for_status = MagicMock()
            response.content = b""
            response.text = ""
            return response

        # Create context
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="openrouter_llm",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )

        # Process batch of rows
        rows = [{"text": f"row_{i}"} for i in range(5)]
        results = []

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = mock_post
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

            for i, row in enumerate(rows):
                row_rec = recorder.create_row(
                    run_id=run.run_id,
                    source_node_id=node.node_id,
                    row_index=i,
                    data=row,
                )
                token = recorder.create_token(row_id=row_rec.row_id)
                state = recorder.begin_node_state(
                    token_id=token.token_id,
                    node_id=node.node_id,
                    step_index=0,
                    input_data=row,
                )

                ctx = PluginContext(
                    run_id=run.run_id,
                    config={},
                    landscape=recorder,
                    state_id=state.state_id,
                )

                result = transform.process(row, ctx)
                results.append(result)

        # All should succeed (capacity errors were retried)
        assert all(r.status == "success" for r in results)
        assert len(results) == 5

        transform.close()
```

### Step 2: Run test

Run: `pytest tests/integration/test_llm_transforms.py::TestPooledExecutionIntegration -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/integration/test_llm_transforms.py
git commit -m "test(llm): add pooled execution integration test"
```

---

## Summary

| Task | Component | Purpose |
|------|-----------|---------|
| 1-4 | `AIMDThrottle` | TCP-style congestion control |
| 5 | `capacity_errors` | Error classification |
| 6 | `ReorderBuffer` | Strict output ordering |
| 7 | `PoolConfig` | Configuration schema |
| 8-10 | `PooledExecutor` | Parallel execution engine |
| 11 | `OpenRouterLLMTransform` | Integration |
| 12 | Property tests | Ordering invariant |
| 13 | Module exports | API surface |
| 14 | Integration test | End-to-end validation |

---

## Post-Implementation

After all tasks complete:

1. **Run full test suite:** `pytest tests/ -v`
2. **Run type checking:** `mypy src/`
3. **Run linting:** `ruff check src/`
4. **Update CHANGELOG:** Add entry for pooled LLM queries
5. **Consider:** Azure transform integration (similar pattern)
