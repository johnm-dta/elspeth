# Pooling Validation and Concurrency Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two pooling subsystem bugs: (1) missing min/max delay invariant validation in PoolConfig, and (2) shared ReorderBuffer allowing result mixing in concurrent execute_batch() calls.

**Architecture:** Both fixes are defensive guards. Bug 1 adds Pydantic model validation to reject invalid configs at construction time. Bug 2 adds a threading.Lock to serialize execute_batch() calls, enforcing single-flight semantics (matching current engine usage patterns).

**Tech Stack:** Pydantic model validators, Python threading.Lock, pytest

---

## Task 1: Add min/max delay invariant validation to PoolConfig

**Files:**
- Modify: `src/elspeth/plugins/pooling/config.py:1-40`
- Test: `tests/plugins/llm/test_pool_config.py`

**Step 1: Write the failing test for min > max rejection**

Add to `tests/plugins/llm/test_pool_config.py` in the `TestPoolConfigValidation` class:

```python
def test_min_dispatch_delay_must_not_exceed_max(self) -> None:
    """min_dispatch_delay_ms must be <= max_dispatch_delay_ms."""
    from pydantic import ValidationError

    from elspeth.plugins.pooling import PoolConfig

    with pytest.raises(ValidationError) as exc_info:
        PoolConfig(
            pool_size=10,
            min_dispatch_delay_ms=1000,
            max_dispatch_delay_ms=100,
        )

    # Verify the error message mentions the invariant
    error_str = str(exc_info.value)
    assert "min_dispatch_delay_ms" in error_str or "cannot exceed" in error_str.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_pool_config.py::TestPoolConfigValidation::test_min_dispatch_delay_must_not_exceed_max -v`

Expected: FAIL - the PoolConfig is created without error (no validation exists)

**Step 3: Write the validator in PoolConfig**

Modify `src/elspeth/plugins/pooling/config.py`:

```python
# src/elspeth/plugins/pooling/config.py
"""Pool configuration for concurrent API transforms."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

from elspeth.plugins.pooling.throttle import ThrottleConfig


class PoolConfig(BaseModel):
    """Pool configuration for concurrent API requests.

    Attributes:
        pool_size: Number of concurrent requests (must be >= 1)
        min_dispatch_delay_ms: Floor for delay between dispatches
        max_dispatch_delay_ms: Ceiling for delay
        backoff_multiplier: Multiply delay on capacity error (must be > 1)
        recovery_step_ms: Subtract from delay on success
        max_capacity_retry_seconds: Max time to retry capacity errors per row
    """

    model_config = {"extra": "forbid"}

    pool_size: int = Field(1, ge=1, description="Number of concurrent requests")
    min_dispatch_delay_ms: int = Field(0, ge=0, description="Minimum dispatch delay in milliseconds")
    max_dispatch_delay_ms: int = Field(5000, ge=0, description="Maximum dispatch delay in milliseconds")
    backoff_multiplier: float = Field(2.0, gt=1.0, description="Backoff multiplier on capacity error")
    recovery_step_ms: int = Field(50, ge=0, description="Recovery step in milliseconds")
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")

    @model_validator(mode="after")
    def _validate_delay_invariants(self) -> Self:
        """Ensure min_dispatch_delay_ms <= max_dispatch_delay_ms."""
        if self.min_dispatch_delay_ms > self.max_dispatch_delay_ms:
            raise ValueError(
                f"min_dispatch_delay_ms ({self.min_dispatch_delay_ms}) cannot exceed "
                f"max_dispatch_delay_ms ({self.max_dispatch_delay_ms})"
            )
        return self

    def to_throttle_config(self) -> ThrottleConfig:
        """Convert to ThrottleConfig for runtime use."""
        return ThrottleConfig(
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/llm/test_pool_config.py::TestPoolConfigValidation::test_min_dispatch_delay_must_not_exceed_max -v`

Expected: PASS

**Step 5: Write edge case test for min == max (should be allowed)**

Add to `tests/plugins/llm/test_pool_config.py` in the `TestPoolConfigValidation` class:

```python
def test_min_equal_to_max_dispatch_delay_is_allowed(self) -> None:
    """min_dispatch_delay_ms == max_dispatch_delay_ms should be allowed (fixed delay)."""
    from elspeth.plugins.pooling import PoolConfig

    # This should NOT raise - equal values are valid (fixed delay)
    config = PoolConfig(
        pool_size=10,
        min_dispatch_delay_ms=500,
        max_dispatch_delay_ms=500,
    )

    assert config.min_dispatch_delay_ms == 500
    assert config.max_dispatch_delay_ms == 500
```

**Step 6: Run edge case test**

Run: `pytest tests/plugins/llm/test_pool_config.py::TestPoolConfigValidation::test_min_equal_to_max_dispatch_delay_is_allowed -v`

Expected: PASS

**Step 7: Run all pool config tests to ensure no regressions**

Run: `pytest tests/plugins/llm/test_pool_config.py -v`

Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/elspeth/plugins/pooling/config.py tests/plugins/llm/test_pool_config.py
git commit -m "$(cat <<'EOF'
fix(pooling): validate min_dispatch_delay_ms <= max_dispatch_delay_ms

Add Pydantic model validator to PoolConfig that rejects configurations
where min_dispatch_delay_ms exceeds max_dispatch_delay_ms. This prevents
confusing AIMD throttle behavior where the "floor" exceeds the "ceiling".

Closes: P3-2026-01-21-pooling-delay-invariant-not-validated

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add single-flight lock to execute_batch() to prevent result mixing

**Files:**
- Modify: `src/elspeth/plugins/pooling/executor.py:80-215`
- Test: `tests/plugins/llm/test_pooled_executor.py`

**Step 1: Write the failing test for concurrent execute_batch isolation**

Add to `tests/plugins/llm/test_pooled_executor.py`:

```python
class TestPooledExecutorConcurrentBatches:
    """Test that concurrent execute_batch() calls are properly isolated."""

    def test_concurrent_execute_batch_calls_are_serialized(self) -> None:
        """Concurrent execute_batch() calls must not mix results.

        This test verifies that if two threads call execute_batch() on the
        same executor, results are not interleaved between batches.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        config = PoolConfig(pool_size=4)
        executor = PooledExecutor(config)

        batch_a_results: list[TransformResult] = []
        batch_b_results: list[TransformResult] = []
        errors: list[Exception] = []

        def process_a(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.02)  # Simulate work
            return TransformResult.success({"batch": "A", "idx": row["idx"]})

        def process_b(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.02)  # Simulate work
            return TransformResult.success({"batch": "B", "idx": row["idx"]})

        def run_batch_a() -> None:
            try:
                contexts = [
                    RowContext(row={"idx": i}, state_id=f"a_state_{i}", row_index=i)
                    for i in range(5)
                ]
                nonlocal batch_a_results
                batch_a_results = executor.execute_batch(contexts, process_a)
            except Exception as e:
                errors.append(e)

        def run_batch_b() -> None:
            try:
                contexts = [
                    RowContext(row={"idx": i}, state_id=f"b_state_{i}", row_index=i)
                    for i in range(5)
                ]
                nonlocal batch_b_results
                batch_b_results = executor.execute_batch(contexts, process_b)
            except Exception as e:
                errors.append(e)

        # Run both batches concurrently
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(run_batch_a)
            future_b = pool.submit(run_batch_b)
            future_a.result(timeout=10)
            future_b.result(timeout=10)

        # Check for errors
        assert not errors, f"Batch execution raised errors: {errors}"

        # Both batches should have exactly 5 results
        assert len(batch_a_results) == 5, f"Batch A has {len(batch_a_results)} results, expected 5"
        assert len(batch_b_results) == 5, f"Batch B has {len(batch_b_results)} results, expected 5"

        # All batch A results must be from batch A (no mixing)
        for i, result in enumerate(batch_a_results):
            assert result.row is not None
            assert result.row["batch"] == "A", f"Batch A result {i} contains batch {result.row['batch']}"
            assert result.row["idx"] == i, f"Batch A result {i} has wrong index {result.row['idx']}"

        # All batch B results must be from batch B (no mixing)
        for i, result in enumerate(batch_b_results):
            assert result.row is not None
            assert result.row["batch"] == "B", f"Batch B result {i} contains batch {result.row['batch']}"
            assert result.row["idx"] == i, f"Batch B result {i} has wrong index {result.row['idx']}"

        executor.shutdown()
```

**Step 2: Run test to verify it fails (or exhibits flaky behavior)**

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorConcurrentBatches::test_concurrent_execute_batch_calls_are_serialized -v`

Expected: FAIL or flaky - results may be mixed between batches due to shared ReorderBuffer

**Note:** This test may pass sometimes due to timing. The bug is a race condition. To reliably reproduce, you may need to run the test multiple times: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorConcurrentBatches::test_concurrent_execute_batch_calls_are_serialized -v --count=10` (requires pytest-repeat plugin, or just run manually several times).

**Step 3: Add the execute_batch lock to PooledExecutor**

Modify `src/elspeth/plugins/pooling/executor.py`:

1. Add `_batch_lock: Lock` to `__init__`:

```python
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

    # Lock to serialize execute_batch calls (single-flight)
    # This prevents concurrent batches from mixing results in the shared buffer
    self._batch_lock = Lock()

    self._shutdown = False
```

2. Add lock acquisition at the start of `execute_batch`:

```python
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

    Note: This method is serialized - only one batch can execute at a time.
    Concurrent calls will block until the previous batch completes.

    Args:
        contexts: List of RowContext with row data and state_ids
        process_fn: Function that processes a single row with state_id

    Returns:
        List of TransformResults in same order as input contexts
    """
    if not contexts:
        return []

    # Serialize batch execution to prevent result mixing
    # The ReorderBuffer uses sequential indices, so concurrent batches
    # would interleave indices and cause results to be returned to the wrong caller
    with self._batch_lock:
        return self._execute_batch_locked(contexts, process_fn)
```

3. Extract the existing batch logic into `_execute_batch_locked`:

```python
def _execute_batch_locked(
    self,
    contexts: list[RowContext],
    process_fn: Callable[[dict[str, Any], str], TransformResult],
) -> list[TransformResult]:
    """Internal batch execution (must be called while holding _batch_lock).

    Args:
        contexts: List of RowContext with row data and state_ids
        process_fn: Function that processes a single row with state_id

    Returns:
        List of TransformResults in same order as input contexts
    """
    # Track futures by their buffer index
    futures: dict[Future[tuple[int, TransformResult]], int] = {}

    # Submit all rows
    for ctx in contexts:
        # Reserve slot in reorder buffer
        buffer_idx = self._buffer.submit()

        # Submit to thread pool
        # NOTE: Semaphore is acquired INSIDE the worker, not here.
        # This prevents deadlock when capacity errors cause workers to
        # release-then-reacquire: if we acquired here, the main thread
        # could steal permits for queued tasks that can't run because
        # worker threads are blocked waiting to reacquire.
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
```

**Step 4: Update imports**

Add `Lock` to the imports at the top of `executor.py`:

```python
from threading import Lock, Semaphore
```

**Step 5: Run the concurrent test to verify it passes**

Run: `pytest tests/plugins/llm/test_pooled_executor.py::TestPooledExecutorConcurrentBatches::test_concurrent_execute_batch_calls_are_serialized -v`

Expected: PASS

**Step 6: Run all executor tests to ensure no regressions**

Run: `pytest tests/plugins/llm/test_pooled_executor.py -v`

Expected: All tests PASS

**Step 7: Run entire pooling test suite**

Run: `pytest tests/plugins/llm/test_pool*.py tests/plugins/llm/test_reorder_buffer.py -v`

Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/elspeth/plugins/pooling/executor.py tests/plugins/llm/test_pooled_executor.py
git commit -m "$(cat <<'EOF'
fix(pooling): serialize execute_batch to prevent concurrent result mixing

Add threading.Lock to PooledExecutor that serializes execute_batch() calls.
This prevents the shared ReorderBuffer from mixing results when concurrent
batches interleave buffer indices.

The fix enforces single-flight semantics, matching the current engine's
sequential transform execution pattern. Future parallel transform support
would need per-batch buffer isolation instead.

Closes: P3-2026-01-21-pooling-concurrent-execute-batch-mixes-results

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Final verification and bug closure

**Step 1: Run the full test suite for pooling**

Run: `pytest tests/plugins/llm/test_pool*.py tests/plugins/llm/test_reorder_buffer.py tests/plugins/llm/test_aimd_throttle.py -v`

Expected: All tests PASS

**Step 2: Run mypy type check on modified files**

Run: `mypy src/elspeth/plugins/pooling/config.py src/elspeth/plugins/pooling/executor.py`

Expected: No errors

**Step 3: Run ruff lint check**

Run: `ruff check src/elspeth/plugins/pooling/config.py src/elspeth/plugins/pooling/executor.py`

Expected: No errors

**Step 4: Move bug reports to closed directory**

```bash
mkdir -p docs/bugs/closed/engine-pooling
mv docs/bugs/open/engine-pooling/P3-2026-01-21-pooling-delay-invariant-not-validated.md docs/bugs/closed/engine-pooling/
mv docs/bugs/open/engine-pooling/P3-2026-01-21-pooling-concurrent-execute-batch-mixes-results.md docs/bugs/closed/engine-pooling/
git add docs/bugs/
git commit -m "docs(bugs): close 2 pooling validation/concurrency bugs"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/elspeth/plugins/pooling/config.py` | Add `@model_validator` for min/max delay invariant |
| `src/elspeth/plugins/pooling/executor.py` | Add `_batch_lock` and serialize `execute_batch()` |
| `tests/plugins/llm/test_pool_config.py` | Add 2 validation tests |
| `tests/plugins/llm/test_pooled_executor.py` | Add concurrent batch isolation test |

## Remaining Open Bugs (Not Addressed)

These bugs remain open in `docs/bugs/open/engine-pooling/`:
- **P2-pooling-ordering-metadata-dropped** - Requires design decision on audit storage
- **P3-pooling-missing-pool-stats** - Feature incomplete, lower priority
- **P2-pooling-throttle-dispatch-burst** - Complex fix requiring dispatcher restructuring
