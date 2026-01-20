# Pooled LLM Queries Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable parallel LLM API calls within a single transform while maintaining strict row order and gracefully handling capacity errors with AIMD throttling.

**Architecture:** Per-transform `PooledExecutor` manages a semaphore-controlled dispatch queue, AIMD throttle for adaptive rate control, and reorder buffer for strict output ordering. The executor wraps HTTP calls from existing `AuditedHTTPClient` and is synchronous from the caller's perspective.

**Tech Stack:** `concurrent.futures.ThreadPoolExecutor`, `threading.Semaphore`, existing `AuditedHTTPClient`, Pydantic config models, pytest/Hypothesis for testing.

**Design Document:** See `docs/plans/2026-01-20-pooled-llm-queries-design.md` for full specification.

---

## Code Review Findings (Addressed in This Plan)

The following issues from code review have been addressed:

| Issue | Resolution |
|-------|------------|
| **Critical:** No `state_id` per-row in pooled execution | Added `RowContext` dataclass to pass `state_id` per row (Task 9) |
| **Critical:** Infinite retry with no escape | Added `max_capacity_retry_seconds` config (Task 7) |
| **Critical:** `_call_index` in AuditedClientBase not thread-safe | Added `Lock` to `_next_call_index()` (Task 0 - NEW) |
| **Critical:** Throttle delay before semaphore acquire | Moved delay inside worker, after semaphore acquire (Task 9) |
| **Critical:** Semaphore held during capacity retry loop | Release before sleep, re-acquire after (Task 10) |
| **Important:** Incomplete buffer drain after `as_completed` | Added final drain loop (Task 9) |
| Hypothesis test uses `random.shuffle` | Fixed to use `st.permutations()` (Task 12) |
| Wrong class name `AzureOpenAILLMTransform` | Fixed to `AzureLLMTransform` (Task 13) |
| Module exports internal classes | Only export public API, keep internals private (Task 13) |
| Missing timing in reorder buffer | Added `submit_timestamp`/`complete_timestamp` (Task 6) |

---

## Task 0: Thread-Safe Call Index in AuditedClientBase

**Files:**
- Modify: `src/elspeth/plugins/clients/base.py`
- Test: `tests/plugins/clients/test_audited_client_base.py`

**CRITICAL FIX:** The existing `AuditedClientBase._next_call_index()` is not thread-safe. With pooled execution, multiple threads calling this simultaneously will cause `call_index` collisions, corrupting the audit trail.

### Step 1: Write failing test for thread safety

```python
# tests/plugins/clients/test_audited_client_base.py
"""Tests for AuditedClientBase thread safety."""

import threading
from unittest.mock import MagicMock

import pytest

from elspeth.plugins.clients.base import AuditedClientBase


class ConcreteAuditedClient(AuditedClientBase):
    """Concrete implementation for testing."""
    pass


class TestCallIndexThreadSafety:
    """Test that _next_call_index is thread-safe."""

    def test_concurrent_call_index_no_duplicates(self) -> None:
        """Multiple threads should get unique call indices."""
        mock_recorder = MagicMock()
        client = ConcreteAuditedClient(
            recorder=mock_recorder,
            state_id="test-state",
        )

        indices: list[int] = []
        lock = threading.Lock()

        def get_indices(count: int) -> None:
            for _ in range(count):
                idx = client._next_call_index()
                with lock:
                    indices.append(idx)

        # Spawn 10 threads, each getting 100 indices
        threads = [
            threading.Thread(target=get_indices, args=(100,))
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 1000 indices should be unique
        assert len(indices) == 1000
        assert len(set(indices)) == 1000, "Duplicate call indices detected!"

        # Should be 0-999
        assert sorted(indices) == list(range(1000))
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/clients/test_audited_client_base.py::TestCallIndexThreadSafety -v`
Expected: FAIL with duplicate indices (race condition)

### Step 3: Add thread-safe locking

```python
# Modify src/elspeth/plugins/clients/base.py

# Add import at top:
from threading import Lock

# In AuditedClientBase.__init__, add:
        self._call_index_lock = Lock()

# Replace _next_call_index method:
    def _next_call_index(self) -> int:
        """Get next call index for this client (thread-safe).

        Returns:
            Sequential call index, unique within this client instance
        """
        with self._call_index_lock:
            idx = self._call_index
            self._call_index += 1
            return idx
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/clients/test_audited_client_base.py::TestCallIndexThreadSafety -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/clients/base.py tests/plugins/clients/test_audited_client_base.py
git commit -m "fix(clients): make AuditedClientBase._next_call_index thread-safe"
```

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

    Note: This is a runtime dataclass, not a Pydantic model, because it's
    internal state configuration built from validated PoolConfig, not
    user-provided YAML config.

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

    def test_stats_track_total_throttle_time(self) -> None:
        """Stats should track total time spent throttled."""
        throttle = AIMDThrottle()

        # Record some throttle time manually (simulating waits)
        throttle.record_throttle_wait(100.0)
        throttle.record_throttle_wait(50.0)

        stats = throttle.get_stats()
        assert stats["total_throttle_time_ms"] == 150.0

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
        self._total_throttle_time_ms: float = 0.0

# Update on_capacity_error to track stats (add inside the lock):
            self._capacity_retries += 1
            if self._current_delay_ms > self._peak_delay_ms:
                self._peak_delay_ms = self._current_delay_ms

# Update on_success to track stats (add inside the lock):
            self._successes += 1

# Add new methods:
    def record_throttle_wait(self, wait_ms: float) -> None:
        """Record time spent waiting due to throttle (thread-safe).

        Args:
            wait_ms: Milliseconds spent waiting
        """
        with self._lock:
            self._total_throttle_time_ms += wait_ms

    def get_stats(self) -> dict[str, float | int]:
        """Get throttle statistics for audit trail (thread-safe).

        Returns:
            Dict with capacity_retries, successes, peak_delay_ms, current_delay_ms,
            total_throttle_time_ms
        """
        with self._lock:
            return {
                "capacity_retries": self._capacity_retries,
                "successes": self._successes,
                "peak_delay_ms": self._peak_delay_ms,
                "current_delay_ms": self._current_delay_ms,
                "total_throttle_time_ms": self._total_throttle_time_ms,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters (thread-safe).

        Note: Does NOT reset current_delay - only resets counters.
        """
        with self._lock:
            self._capacity_retries = 0
            self._successes = 0
            self._peak_delay_ms = self._current_delay_ms
            self._total_throttle_time_ms = 0.0
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
with AIMD throttling. They are distinct from "normal" errors
(auth failures, malformed requests) which use standard retry limits.

HTTP Status Codes:
- 429: Too Many Requests (universal)
- 503: Service Unavailable (universal)
- 529: Overloaded (Azure, some other providers)
"""

from __future__ import annotations


# HTTP status codes that indicate capacity/rate limiting
# These trigger AIMD throttle and capacity retry
CAPACITY_ERROR_CODES: frozenset[int] = frozenset({429, 503, 529})


def is_capacity_error(status_code: int) -> bool:
    """Check if HTTP status code indicates a capacity error.

    Capacity errors are transient overload conditions that should trigger
    AIMD throttle backoff and be retried with increasing delays.

    Args:
        status_code: HTTP status code

    Returns:
        True if this is a capacity error, False otherwise
    """
    return status_code in CAPACITY_ERROR_CODES


class CapacityError(Exception):
    """Exception for capacity/rate limit errors.

    Raised when an LLM API call fails due to capacity limits.
    These errors trigger AIMD throttle and are retried until
    max_capacity_retry_seconds is exceeded.

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

## Task 6: Reorder Buffer with Timing

**Files:**
- Create: `src/elspeth/plugins/llm/reorder_buffer.py`
- Test: `tests/plugins/llm/test_reorder_buffer.py`

Buffer that maintains strict submission order with timing for audit.

### Step 1: Write failing test for buffer

```python
# tests/plugins/llm/test_reorder_buffer.py
"""Tests for reorder buffer that maintains submission order."""

import time

import pytest

from elspeth.plugins.llm.reorder_buffer import ReorderBuffer, BufferEntry


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
        assert len(results) == 1
        assert results[0].result == "result_0"
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
        ready = buffer.get_ready_results()
        assert len(ready) == 1
        assert ready[0].result == "result_0"

        buffer.complete(4, "result_4")
        assert buffer.get_ready_results() == []  # Still waiting for 1

        buffer.complete(1, "result_1")
        # Now 1 and 2 can be emitted
        ready = buffer.get_ready_results()
        assert len(ready) == 2
        assert ready[0].result == "result_1"
        assert ready[1].result == "result_2"

        buffer.complete(3, "result_3")
        # Now 3 and 4 can be emitted
        ready = buffer.get_ready_results()
        assert len(ready) == 2
        assert ready[0].result == "result_3"
        assert ready[1].result == "result_4"

        assert buffer.pending_count == 0

    def test_in_order_completion_immediate(self) -> None:
        """Results completing in order should emit immediately."""
        buffer = ReorderBuffer[str]()

        for i in range(3):
            idx = buffer.submit()
            buffer.complete(idx, f"result_{i}")
            ready = buffer.get_ready_results()
            assert len(ready) == 1
            assert ready[0].result == f"result_{i}"

        assert buffer.pending_count == 0


class TestReorderBufferTiming:
    """Test timing metadata for audit trail."""

    def test_entry_has_submit_timestamp(self) -> None:
        """Buffer entries should record submit timestamp."""
        buffer = ReorderBuffer[str]()

        before = time.perf_counter()
        idx = buffer.submit()
        after = time.perf_counter()

        buffer.complete(idx, "result")
        ready = buffer.get_ready_results()

        assert len(ready) == 1
        assert before <= ready[0].submit_timestamp <= after

    def test_entry_has_complete_timestamp(self) -> None:
        """Buffer entries should record complete timestamp."""
        buffer = ReorderBuffer[str]()

        idx = buffer.submit()
        time.sleep(0.01)  # Small delay

        before = time.perf_counter()
        buffer.complete(idx, "result")
        after = time.perf_counter()

        ready = buffer.get_ready_results()

        assert len(ready) == 1
        assert before <= ready[0].complete_timestamp <= after
        # Complete should be after submit
        assert ready[0].complete_timestamp > ready[0].submit_timestamp

    def test_entry_tracks_buffer_wait_time(self) -> None:
        """Entry should track time spent waiting in buffer."""
        buffer = ReorderBuffer[str]()

        # Submit two items
        idx0 = buffer.submit()
        idx1 = buffer.submit()

        # Complete second first (will wait in buffer)
        buffer.complete(idx1, "result_1")
        time.sleep(0.02)  # Wait while 1 is buffered

        # Complete first (releases both)
        buffer.complete(idx0, "result_0")
        ready = buffer.get_ready_results()

        assert len(ready) == 2
        # First item shouldn't have waited much
        assert ready[0].buffer_wait_ms < 50
        # Second item waited while first was pending
        assert ready[1].buffer_wait_ms >= 15  # At least 20ms minus some tolerance
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_reorder_buffer.py -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement reorder buffer with timing

```python
# src/elspeth/plugins/llm/reorder_buffer.py
"""Reorder buffer for maintaining strict submission order with timing.

Results may complete out of order (due to varying API latencies),
but are emitted in the exact order they were submitted. Timing
metadata is captured for audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class BufferEntry(Generic[T]):
    """Entry emitted from the reorder buffer with timing metadata.

    Attributes:
        submit_index: Order in which item was submitted (0-indexed)
        complete_index: Order in which item completed (may differ from submit)
        result: The actual result value
        submit_timestamp: time.perf_counter() when submitted
        complete_timestamp: time.perf_counter() when completed
        buffer_wait_ms: Time spent waiting in buffer after completion
    """

    submit_index: int
    complete_index: int
    result: T
    submit_timestamp: float
    complete_timestamp: float
    buffer_wait_ms: float


@dataclass
class _InternalEntry(Generic[T]):
    """Internal entry in the reorder buffer."""

    submit_index: int
    submit_timestamp: float
    complete_index: int | None = None
    complete_timestamp: float | None = None
    result: T | None = None
    is_complete: bool = False


class ReorderBuffer(Generic[T]):
    """Thread-safe buffer that reorders results to match submission order.

    Captures timing metadata for each entry to support audit trail
    requirements.

    Usage:
        buffer = ReorderBuffer[TransformResult]()

        # Submit work (returns index)
        idx = buffer.submit()

        # ... do async work ...

        # Complete with result (may be out of order)
        buffer.complete(idx, result)

        # Get results in submission order with timing
        ready = buffer.get_ready_results()
        for entry in ready:
            print(f"Result: {entry.result}, waited {entry.buffer_wait_ms}ms")
    """

    def __init__(self) -> None:
        """Initialize empty buffer."""
        self._entries: dict[int, _InternalEntry[T]] = {}
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
            self._entries[idx] = _InternalEntry(
                submit_index=idx,
                submit_timestamp=time.perf_counter(),
            )
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
            entry.complete_timestamp = time.perf_counter()
            entry.is_complete = True
            self._complete_counter += 1

    def get_ready_results(self) -> list[BufferEntry[T]]:
        """Get all results that are ready to emit in order (thread-safe).

        Returns results that are:
        1. Complete (result received)
        2. All previous indices are also complete

        Returns:
            List of BufferEntry in submission order (may be empty)
        """
        with self._lock:
            ready: list[BufferEntry[T]] = []
            now = time.perf_counter()

            while self._next_emit in self._entries:
                entry = self._entries[self._next_emit]
                if not entry.is_complete:
                    break

                # Entry is complete and all previous are emitted
                # Calculate buffer wait time (time between completion and emission)
                buffer_wait_ms = (now - entry.complete_timestamp) * 1000  # type: ignore[operator]

                ready.append(
                    BufferEntry(
                        submit_index=entry.submit_index,
                        complete_index=entry.complete_index,  # type: ignore[arg-type]
                        result=entry.result,  # type: ignore[arg-type]
                        submit_timestamp=entry.submit_timestamp,
                        complete_timestamp=entry.complete_timestamp,  # type: ignore[arg-type]
                        buffer_wait_ms=buffer_wait_ms,
                    )
                )
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
git commit -m "feat(llm): add reorder buffer with timing for audit"
```

---

## Task 7: Pool Configuration Schema with Max Retry Timeout

**Files:**
- Modify: `src/elspeth/plugins/llm/base.py`
- Test: `tests/plugins/llm/test_pool_config.py`

Add pool configuration fields to LLMConfig including max retry timeout.

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
        # Max retry timeout default (1 hour)
        assert config.pool_config.max_capacity_retry_seconds == 3600

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
            "max_capacity_retry_seconds": 1800,  # 30 minutes
        })

        assert config.pool_config is not None
        assert config.pool_config.pool_size == 5
        assert config.pool_config.min_dispatch_delay_ms == 10
        assert config.pool_config.max_dispatch_delay_ms == 1000
        assert config.pool_config.backoff_multiplier == 3.0
        assert config.pool_config.recovery_step_ms == 25
        assert config.pool_config.max_capacity_retry_seconds == 1800


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

    def test_max_capacity_retry_seconds_must_be_positive(self) -> None:
        """max_capacity_retry_seconds must be > 0."""
        with pytest.raises(Exception):
            LLMConfig.from_dict({
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": {"fields": "dynamic"},
                "pool_size": 10,
                "max_capacity_retry_seconds": 0,
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

    Note: This is an internal model built from flat LLMConfig fields,
    not directly exposed in YAML. Users configure flat fields like
    `pool_size: 10` on the transform config.
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
    max_capacity_retry_seconds: int = Field(
        3600,  # 1 hour default
        gt=0,
        description="Max seconds to retry capacity errors per row before failing",
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
    max_capacity_retry_seconds: int = Field(
        3600, gt=0, description="Max seconds to retry capacity errors per row"
    )

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
            max_capacity_retry_seconds=self.max_capacity_retry_seconds,
        )
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_pool_config.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/base.py tests/plugins/llm/test_pool_config.py
git commit -m "feat(llm): add pool configuration with max retry timeout"
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

from typing import Any

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

        executor.shutdown()

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

        executor.shutdown()


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
- Enforcing max retry timeout for capacity errors
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Semaphore
from typing import TYPE_CHECKING, Any, Callable

from elspeth.contracts import TransformResult
from elspeth.plugins.llm.aimd_throttle import AIMDThrottle
from elspeth.plugins.llm.base import PoolConfig
from elspeth.plugins.llm.reorder_buffer import ReorderBuffer, BufferEntry

if TYPE_CHECKING:
    pass


@dataclass
class RowContext:
    """Context for processing a single row in the pool.

    This allows each row to have its own state_id for audit trail,
    solving the "single state_id for all parallel rows" problem.

    Attributes:
        row: The row data to process
        state_id: Unique state ID for this row's audit trail
        row_index: Original index for ordering
    """

    row: dict[str, Any]
    state_id: str
    row_index: int


class PooledExecutor:
    """Executor for parallel LLM API calls with strict ordering.

    Manages a pool of concurrent requests with:
    - Semaphore-controlled dispatch (max pool_size in flight)
    - AIMD throttle for adaptive rate limiting
    - Reorder buffer for strict submission order output
    - Max retry timeout for capacity errors

    The executor is synchronous from the caller's perspective -
    execute_batch() blocks until all results are ready in order.

    Usage:
        executor = PooledExecutor(pool_config)

        # Prepare row contexts with per-row state IDs
        contexts = [
            RowContext(row=row, state_id=state_id, row_index=i)
            for i, (row, state_id) in enumerate(zip(rows, state_ids))
        ]

        # Process batch
        results = executor.execute_batch(
            contexts=contexts,
            process_fn=lambda row, state_id: transform.process_single(row, state_id),
        )

        # Results are in submission order
        assert len(results) == len(contexts)

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
        self._max_capacity_retry_seconds = config.max_capacity_retry_seconds

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
                "max_capacity_retry_seconds": self._max_capacity_retry_seconds,
            },
            "pool_stats": {
                "capacity_retries": throttle_stats["capacity_retries"],
                "successes": throttle_stats["successes"],
                "peak_delay_ms": throttle_stats["peak_delay_ms"],
                "current_delay_ms": throttle_stats["current_delay_ms"],
                "total_throttle_time_ms": throttle_stats["total_throttle_time_ms"],
            },
        }
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_pooled_executor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/pooled_executor.py tests/plugins/llm/test_pooled_executor.py
git commit -m "feat(llm): add PooledExecutor core structure with RowContext"
```

---

## Task 9: PooledExecutor Batch Execution with Per-Row State

**Files:**
- Modify: `src/elspeth/plugins/llm/pooled_executor.py`
- Modify: `tests/plugins/llm/test_pooled_executor.py`

Implement the main execute_batch method with per-row state_id handling.

### Step 1: Write failing test for batch execution

```python
# Add to tests/plugins/llm/test_pooled_executor.py

import time
from threading import Lock

class TestPooledExecutorBatch:
    """Test batch execution with ordering."""

    def test_execute_batch_returns_results_in_order(self) -> None:
        """Results should be in submission order regardless of completion."""
        config = PoolConfig(pool_size=3)
        executor = PooledExecutor(config)

        # Mock process function with varying delays
        call_order: list[int] = []
        lock = Lock()

        def mock_process(row: dict, state_id: str) -> TransformResult:
            idx = row["idx"]
            with lock:
                call_order.append(idx)
            # Varying delays to cause out-of-order completion
            time.sleep(0.01 * (3 - idx))  # idx 0 slowest, idx 2 fastest
            return TransformResult.success({"idx": idx, "result": f"done_{idx}"})

        contexts = [
            RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i)
            for i in range(3)
        ]

        results = executor.execute_batch(contexts, mock_process)

        # Results must be in submission order
        assert len(results) == 3
        assert results[0].row["idx"] == 0
        assert results[1].row["idx"] == 1
        assert results[2].row["idx"] == 2

        executor.shutdown()

    def test_execute_batch_passes_state_id_per_row(self) -> None:
        """Each row should receive its own state_id."""
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        received_state_ids: list[tuple[int, str]] = []
        lock = Lock()

        def mock_process(row: dict, state_id: str) -> TransformResult:
            with lock:
                received_state_ids.append((row["idx"], state_id))
            return TransformResult.success(row)

        contexts = [
            RowContext(row={"idx": i}, state_id=f"unique_state_{i}", row_index=i)
            for i in range(3)
        ]

        executor.execute_batch(contexts, mock_process)

        # Verify each row got its own state_id
        assert len(received_state_ids) == 3
        state_id_map = {idx: sid for idx, sid in received_state_ids}
        assert state_id_map[0] == "unique_state_0"
        assert state_id_map[1] == "unique_state_1"
        assert state_id_map[2] == "unique_state_2"

        executor.shutdown()

    def test_execute_batch_respects_pool_size(self) -> None:
        """Should never exceed pool_size concurrent requests."""
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        max_concurrent = 0
        current_concurrent = 0
        lock = Lock()

        def mock_process(row: dict, state_id: str) -> TransformResult:
            nonlocal max_concurrent, current_concurrent
            with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent

            time.sleep(0.05)

            with lock:
                current_concurrent -= 1

            return TransformResult.success(row)

        contexts = [
            RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i)
            for i in range(5)
        ]

        results = executor.execute_batch(contexts, mock_process)

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
        contexts: list[RowContext],
        process_fn: Callable[[dict[str, Any], str], TransformResult],
    ) -> list[TransformResult]:
        """Execute batch of rows with parallel processing.

        Dispatches rows to the thread pool with semaphore control,
        applies AIMD throttle delays, and returns results in
        submission order.

        Each row is processed with its own state_id for audit trail.

        Args:
            contexts: List of RowContext with row data and state_ids
            process_fn: Function that processes a single row with state_id

        Returns:
            List of TransformResults in same order as input contexts
        """
        if not contexts:
            return []

        # Track futures by their buffer index
        futures: dict[Future[tuple[int, TransformResult]], int] = {}

        # Submit all rows
        for ctx in contexts:
            # Reserve slot in reorder buffer
            buffer_idx = self._buffer.submit()

            # Acquire semaphore (blocks if pool is full)
            # NOTE: Throttle delay is applied INSIDE the worker, not here,
            # to avoid serial delays blocking parallel submission
            self._semaphore.acquire()

            # Submit to thread pool
            future = self._thread_pool.submit(
                self._execute_single,
                buffer_idx,
                ctx.row,
                ctx.state_id,
                process_fn,
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
            for entry in ready:
                results.append(entry.result)

        # CRITICAL: Final drain - collect any remaining results not yet emitted
        # (the last completed future may not have been at the head of the queue)
        while self._buffer.pending_count > 0:
            ready = self._buffer.get_ready_results()
            if not ready:
                break  # Safety: shouldn't happen if all futures completed
            for entry in ready:
                results.append(entry.result)

        return results

    def _execute_single(
        self,
        buffer_idx: int,
        row: dict[str, Any],
        state_id: str,
        process_fn: Callable[[dict[str, Any], str], TransformResult],
    ) -> tuple[int, TransformResult]:
        """Execute single row and handle throttle feedback.

        Throttle delay is applied HERE (inside the worker) rather than
        in the submission loop. This ensures parallel dispatch isn't
        serialized by throttle delays.

        Args:
            buffer_idx: Index in reorder buffer
            row: Row to process
            state_id: State ID for audit trail
            process_fn: Processing function

        Returns:
            Tuple of (buffer_idx, result)
        """
        try:
            # Apply throttle delay INSIDE worker (after semaphore acquired)
            delay_ms = self._throttle.current_delay_ms
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
                self._throttle.record_throttle_wait(delay_ms)

            result = process_fn(row, state_id)
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
git commit -m "feat(llm): add PooledExecutor batch execution with per-row state"
```

---

## Task 10: PooledExecutor Capacity Error Handling with Timeout

**Files:**
- Modify: `src/elspeth/plugins/llm/pooled_executor.py`
- Modify: `tests/plugins/llm/test_pooled_executor.py`

Add capacity error detection with max retry timeout.

### Step 1: Write failing test for capacity handling

```python
# Add to tests/plugins/llm/test_pooled_executor.py

from elspeth.plugins.llm.capacity_errors import CapacityError

class TestPooledExecutorCapacityHandling:
    """Test capacity error handling with AIMD throttle and timeout."""

    def test_capacity_error_triggers_throttle_and_retries(self) -> None:
        """Capacity errors should trigger throttle and retry."""
        config = PoolConfig(pool_size=2, recovery_step_ms=50)
        executor = PooledExecutor(config)

        call_count = 0
        lock = Lock()

        def mock_process(row: dict, state_id: str) -> TransformResult:
            nonlocal call_count
            with lock:
                call_count += 1
                current_count = call_count

            # First call raises capacity error, second succeeds
            if current_count == 1:
                raise CapacityError(429, "Rate limited")
            return TransformResult.success(row)

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        results = executor.execute_batch(contexts, mock_process)

        # Should have retried and succeeded
        assert len(results) == 1
        assert results[0].status == "success"
        assert call_count == 2

        # Throttle should have been triggered
        stats = executor.get_stats()
        assert stats["pool_stats"]["capacity_retries"] == 1

        executor.shutdown()

    def test_capacity_retry_respects_max_timeout(self) -> None:
        """Capacity retries should stop after max_capacity_retry_seconds."""
        config = PoolConfig(
            pool_size=1,
            max_dispatch_delay_ms=100,
            max_capacity_retry_seconds=1,  # Only 1 second
        )
        executor = PooledExecutor(config)

        call_count = 0
        lock = Lock()

        def mock_process(row: dict, state_id: str) -> TransformResult:
            nonlocal call_count
            with lock:
                call_count += 1
            # Always fail with capacity error
            raise CapacityError(503, "Service unavailable")

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        results = executor.execute_batch(contexts, mock_process)

        # Should eventually fail after timeout
        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].reason is not None
        assert "capacity_retry_timeout" in results[0].reason["reason"]

        # Should have made multiple attempts before giving up
        assert call_count > 1

        executor.shutdown()

    def test_normal_error_not_retried(self) -> None:
        """Non-capacity errors should not be retried."""
        config = PoolConfig(pool_size=1)
        executor = PooledExecutor(config)

        def mock_process(row: dict, state_id: str) -> TransformResult:
            # Return error result (not raise CapacityError)
            return TransformResult.error({"reason": "bad_request"})

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        results = executor.execute_batch(contexts, mock_process)

        # Should return error without retry
        assert len(results) == 1
        assert results[0].status == "error"

        executor.shutdown()

    def test_capacity_retry_releases_semaphore_during_backoff(self) -> None:
        """During capacity retry backoff, semaphore should be released.

        This ensures other workers can make progress while one is sleeping.
        CRITICAL: Without this, all workers hitting capacity errors would
        deadlock the pool.
        """
        config = PoolConfig(
            pool_size=2,
            recovery_step_ms=50,
            max_dispatch_delay_ms=100,
        )
        executor = PooledExecutor(config)

        # Track concurrent execution during retry
        row0_in_retry_sleep = threading.Event()
        row1_completed = threading.Event()
        execution_order: list[str] = []
        lock = Lock()

        def mock_process(row: dict, state_id: str) -> TransformResult:
            idx = row["idx"]
            with lock:
                execution_order.append(f"start_{idx}")

            if idx == 0:
                # Row 0: First call fails, signals it's sleeping, waits, then succeeds
                if not hasattr(mock_process, "row0_called"):
                    mock_process.row0_called = True
                    row0_in_retry_sleep.set()  # Signal we're about to sleep
                    raise CapacityError(429, "Rate limited")
                # Second call succeeds
                row1_completed.wait(timeout=2)  # Wait for row 1 to complete
                with lock:
                    execution_order.append(f"end_{idx}")
                return TransformResult.success(row)
            else:
                # Row 1: Wait until row 0 is in retry sleep, then complete
                row0_in_retry_sleep.wait(timeout=2)
                time.sleep(0.05)  # Give row 0 time to release semaphore
                with lock:
                    execution_order.append(f"end_{idx}")
                row1_completed.set()
                return TransformResult.success(row)

        contexts = [
            RowContext(row={"idx": 0}, state_id="state_0", row_index=0),
            RowContext(row={"idx": 1}, state_id="state_1", row_index=1),
        ]

        results = executor.execute_batch(contexts, mock_process)

        # Both should succeed
        assert len(results) == 2
        assert all(r.status == "success" for r in results)

        # Row 1 should have executed WHILE row 0 was in retry sleep
        # If semaphore wasn't released, row 1 would be blocked
        assert "end_1" in execution_order
        end_1_idx = execution_order.index("end_1")
        # end_1 should happen before end_0 (row 1 completes during row 0's retry)
        assert "end_0" in execution_order
        end_0_idx = execution_order.index("end_0")
        assert end_1_idx < end_0_idx, (
            f"Row 1 should complete before Row 0's retry succeeds. "
            f"Order: {execution_order}"
        )

        executor.shutdown()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorCapacityHandling -v`
Expected: FAIL - capacity errors not handled

### Step 3: Add capacity error handling with timeout

```python
# Modify _execute_single in src/elspeth/plugins/llm/pooled_executor.py

# Add import at top:
from elspeth.plugins.llm.capacity_errors import CapacityError

    def _execute_single(
        self,
        buffer_idx: int,
        row: dict[str, Any],
        state_id: str,
        process_fn: Callable[[dict[str, Any], str], TransformResult],
    ) -> tuple[int, TransformResult]:
        """Execute single row with capacity error retry and timeout.

        Capacity errors trigger AIMD throttle and are retried until
        max_capacity_retry_seconds is exceeded. Normal errors/results
        are returned as-is.
        """
        start_time = time.monotonic()
        max_time = start_time + self._max_capacity_retry_seconds

        try:
            while True:
                try:
                    result = process_fn(row, state_id)
                    self._throttle.on_success()
                    return (buffer_idx, result)
                except CapacityError as e:
                    # Check if we've exceeded max retry time
                    if time.monotonic() >= max_time:
                        elapsed = time.monotonic() - start_time
                        return (
                            buffer_idx,
                            TransformResult.error(
                                {
                                    "reason": "capacity_retry_timeout",
                                    "error": str(e),
                                    "status_code": e.status_code,
                                    "elapsed_seconds": elapsed,
                                    "max_seconds": self._max_capacity_retry_seconds,
                                },
                                retryable=False,
                            ),
                        )

                    # Trigger throttle backoff
                    self._throttle.on_capacity_error()

                    # CRITICAL: Release semaphore BEFORE sleeping
                    # This allows other workers to make progress while we wait
                    self._semaphore.release()

                    # Wait throttle delay before retry
                    delay_ms = self._throttle.current_delay_ms
                    if delay_ms > 0:
                        time.sleep(delay_ms / 1000)
                        self._throttle.record_throttle_wait(delay_ms)

                    # Re-acquire semaphore for retry
                    self._semaphore.acquire()

                    # Retry
                    continue
        finally:
            # Always release semaphore
            # Note: During capacity retry, we release before sleep and re-acquire after,
            # so when we reach this finally block (success, timeout, or other exception),
            # we always hold the semaphore and need to release it exactly once.
            self._semaphore.release()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorCapacityHandling -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/pooled_executor.py tests/plugins/llm/test_pooled_executor.py
git commit -m "feat(llm): add capacity error handling with max retry timeout"
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

        transform.close()
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

# Modify __init__ - add after existing config parsing:
        # Create pooled executor if pool_size > 1
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

# Add method _process_single (extract HTTP logic from process):
    def _process_single_with_state(
        self, row: dict[str, Any], state_id: str
    ) -> TransformResult:
        """Process a single row via OpenRouter API with explicit state_id.

        This is used by the pooled executor where each row has its own state.

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

        # 3. Get recorder from transform's stored reference (set during on_start)
        if self._recorder is None:
            raise RuntimeError(
                "OpenRouter transform requires recorder. "
                "Ensure on_start was called."
            )

        http_client = AuditedHTTPClient(
            recorder=self._recorder,
            state_id=state_id,
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
                raise CapacityError(e.response.status_code, str(e)) from e
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

# Add on_start to capture recorder:
    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder reference for pooled execution."""
        self._recorder = ctx.landscape

# Modify close:
    def close(self) -> None:
        """Release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._recorder = None
```

Also add `self._recorder: LandscapeRecorder | None = None` to `__init__`.

### Step 4: Run test to verify it passes

Run: `pytest tests/integration/test_llm_transforms.py::TestOpenRouterPooledExecution -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/llm/openrouter.py tests/integration/test_llm_transforms.py
git commit -m "feat(llm): integrate PooledExecutor into OpenRouterLLMTransform"
```

---

## Task 12: Property-Based Test for Reorder Buffer (Fixed)

**Files:**
- Modify: `tests/plugins/llm/test_reorder_buffer.py`

Add Hypothesis property test for ordering invariant (fixed to avoid random.shuffle).

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
        for _ in range(n):
            buffer.submit()

        # Complete in permuted order (using Hypothesis-provided permutation)
        for complete_idx in completion_order:
            buffer.complete(complete_idx, complete_idx)

        # Collect all results
        all_results: list[int] = []
        while buffer.pending_count > 0:
            ready = buffer.get_ready_results()
            for entry in ready:
                all_results.append(entry.result)

        # Drain any remaining
        ready = buffer.get_ready_results()
        for entry in ready:
            all_results.append(entry.result)

        # Must be in submission order (0, 1, 2, ..., n-1)
        assert all_results == list(range(n))

    @given(
        completion_order=st.permutations(range(20)),
    )
    @settings(max_examples=50)
    def test_all_submitted_items_eventually_emitted(
        self, completion_order: list[int]
    ) -> None:
        """Every submitted item is eventually emitted exactly once."""
        buffer = ReorderBuffer[str]()
        n = len(completion_order)

        # Submit n items
        for _ in range(n):
            buffer.submit()

        # Complete in Hypothesis-provided permutation order
        for idx in completion_order:
            buffer.complete(idx, f"result_{idx}")

        # Collect all results
        all_results: list[str] = []
        while buffer.pending_count > 0:
            ready = buffer.get_ready_results()
            for entry in ready:
                all_results.append(entry.result)

        # Drain any remaining
        ready = buffer.get_ready_results()
        for entry in ready:
            all_results.append(entry.result)

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

## Task 13: Update Module Exports (Minimal Public API)

**Files:**
- Modify: `src/elspeth/plugins/llm/__init__.py`

Export only public API, keep internal classes private.

### Step 1: Read current exports

Run: `cat src/elspeth/plugins/llm/__init__.py`

### Step 2: Add minimal exports

```python
# src/elspeth/plugins/llm/__init__.py
"""LLM transform plugins for ELSPETH."""

from elspeth.plugins.llm.azure import AzureLLMTransform
from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform
from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig, PoolConfig
from elspeth.plugins.llm.batch_errors import BatchPendingError
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform
from elspeth.plugins.llm.templates import PromptTemplate, RenderedPrompt, TemplateError

__all__ = [
    # Transforms (public)
    "AzureBatchLLMTransform",
    "AzureLLMTransform",
    "BaseLLMTransform",
    "OpenRouterLLMTransform",
    # Config (public - users may inspect)
    "LLMConfig",
    "PoolConfig",
    # Exceptions (public - plugins may catch/raise)
    "BatchPendingError",
    "CapacityError",
    # Templates (public)
    "PromptTemplate",
    "RenderedPrompt",
    "TemplateError",
    # Note: AIMDThrottle, ReorderBuffer, PooledExecutor are internal
    # and not exported. Import directly if needed for testing.
]
```

### Step 3: Run full test suite

Run: `pytest tests/plugins/llm/ -v`
Expected: All tests PASS

### Step 4: Commit

```bash
git add src/elspeth/plugins/llm/__init__.py
git commit -m "feat(llm): export minimal public API for pooled execution"
```

---

## Task 14: Full Integration Test

**Files:**
- Modify: `tests/integration/test_llm_transforms.py`

Add integration test for full pooled execution.

### Step 1: Write integration test

```python
# Add to tests/integration/test_llm_transforms.py

class TestPooledExecutionIntegration:
    """Full integration tests for pooled LLM execution."""

    def test_batch_with_simulated_capacity_errors(self) -> None:
        """Verify pooled execution handles capacity errors correctly."""
        from unittest.mock import patch, MagicMock
        from threading import Lock
        import random

        # Seed random for reproducibility
        random.seed(42)

        # Create transform with pooling
        transform = OpenRouterLLMTransform({
            "model": "test-model",
            "template": "{{ row.text }}",
            "api_key": "test-key",
            "schema": {"fields": "dynamic"},
            "pool_size": 3,
            "max_dispatch_delay_ms": 100,
            "max_capacity_retry_seconds": 10,
        })

        # Track calls per row
        call_counts: dict[int, int] = {}
        lock = Lock()

        def mock_post(*args, **kwargs):
            """Simulate 50% capacity error rate on first call."""
            body = kwargs.get("json", {})
            messages = body.get("messages", [])

            # Extract row index from message
            if messages:
                content = messages[-1].get("content", "")
                idx = int(content.split("_")[1]) if "_" in content else 0
                with lock:
                    call_counts[idx] = call_counts.get(idx, 0) + 1
                    current_count = call_counts[idx]

                # First call has 50% chance of capacity error
                if current_count == 1 and random.random() < 0.5:
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
                "choices": [{"message": {"content": "done"}}],
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

        # Call on_start to set up recorder
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            landscape=recorder,
            state_id=None,  # Will be set per-row
        )
        transform.on_start(ctx)

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

                row_ctx = PluginContext(
                    run_id=run.run_id,
                    config={},
                    landscape=recorder,
                    state_id=state.state_id,
                )

                result = transform.process(row, row_ctx)
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
git commit -m "test(llm): add full pooled execution integration test"
```

---

## Summary

| Task | Component | Purpose |
|------|-----------|---------|
| **0** | `AuditedClientBase` | **Thread-safe `_call_index` with Lock** |
| 1-4 | `AIMDThrottle` | TCP-style congestion control with stats |
| 5 | `capacity_errors` | Error classification |
| 6 | `ReorderBuffer` | Strict output ordering with timing |
| 7 | `PoolConfig` | Configuration schema with max retry timeout |
| 8-10 | `PooledExecutor` | Parallel execution with per-row state |
| 11 | `OpenRouterLLMTransform` | Integration with pooled executor |
| 12 | Property tests | Ordering invariant (Hypothesis) |
| 13 | Module exports | Minimal public API |
| 14 | Integration test | End-to-end validation |

---

## Code Review Issues Addressed

| Issue | How Addressed |
|-------|---------------|
| **CRITICAL:** `_call_index` not thread-safe | Added `Lock` to `_next_call_index()` (Task 0) |
| **CRITICAL:** Throttle delay before semaphore | Moved delay inside worker, after semaphore acquire (Task 9) |
| **CRITICAL:** Semaphore held during retry | Release before sleep, re-acquire after (Task 10) |
| **Important:** Incomplete buffer drain | Added final drain loop after `as_completed` (Task 9) |
| No `state_id` per-row | Added `RowContext` dataclass, `execute_batch` takes list of contexts |
| Infinite retry without timeout | Added `max_capacity_retry_seconds` config (default 1 hour) |
| Hypothesis uses `random.shuffle` | Changed to `st.permutations()` for reproducibility |
| Wrong class name | Fixed to `AzureLLMTransform` |
| Exports internal classes | Only export public API in `__init__.py` |
| Missing timing | Added `submit_timestamp`, `complete_timestamp`, `buffer_wait_ms` to `BufferEntry` |

---

## Post-Implementation

After all tasks complete:

1. **Run full test suite:** `pytest tests/ -v`
2. **Run type checking:** `mypy src/`
3. **Run linting:** `ruff check src/`
4. **Update CHANGELOG:** Add entry for pooled LLM queries
5. **Consider:** Azure transform integration (similar pattern)
