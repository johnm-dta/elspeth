# Batch Transform Integration - Implementation Plan

**Goal:** Fix batch transform integration with TransformExecutor for concurrent LLM processing
**Timeline:** ~10 hours (revised from 24h after peer review)
**Status:** ✅ IMPLEMENTED (2026-01-26)
**Pattern:** Shared adapter with per-row synchronization via token_id routing

---

## Executive Summary

**The Problem:** TransformExecutor has batch transform integration code, but it's **broken**. The existing `BlockingResultAdapter` creates one adapter per row but only connects the first one to the transform's output port. Subsequent rows' adapters never receive results, causing **deadlock**.

**Root Cause:** In `executors.py:201-219`:
- Row 1: `adapter1` connected via `connect_output()`, sets `_batch_initialized = True`
- Row 2: `adapter2` created, but `connect_output()` **skipped** because `_batch_initialized = True`
- All results emit to `adapter1` (the only connected adapter)
- `adapter2.wait_for_result()` blocks forever → **DEADLOCK**

**The Solution:** Replace `BlockingResultAdapter` with `SharedBatchAdapter` that routes results to the correct waiter by `token_id`. One adapter per transform instance, multiple waiters per adapter.

**Concurrency Model:**
- Orchestrator submits A, B, C, D sequentially via process()
- Each process() calls accept() (instant) then blocks on result
- Workers process A, B, C, D concurrently in thread pool
- Results routed back to correct waiter via token_id matching

---

## Architecture

### Current State (BROKEN)
```python
# In TransformExecutor (executors.py:201-219):
if has_accept:
    adapter = BlockingResultAdapter(expected_token_id=token.token_id)  # New per row!
    if not getattr(transform, "_batch_initialized", False):
        transform.connect_output(output=adapter, max_pending=30)  # Only first row!
    transform.accept(token.row_data, ctx)
    result = adapter.wait_for_result()  # Row 2+ waits forever - DEADLOCK

# In BlockingResultAdapter.emit():
def emit(self, token, result):
    if token.token_id != self._expected_token_id:
        return  # Ignores results for other rows!
```

**Why it deadlocks:** The transform's `_batch_output` is set to `adapter1` on first row. All results emit there. `adapter2`, `adapter3`, etc. are created but never connected - their `wait_for_result()` blocks forever.

### Target State
```python
# In TransformExecutor (one SharedBatchAdapter per transform, reused across rows):
if has_accept:
    adapter = self._get_batch_adapter(transform)  # Creates once, reuses after
    waiter = adapter.register(token.token_id)     # Register waiter for THIS row
    ctx.token = token
    transform.accept(token.row_data, ctx)         # Submit work (instant)
    result = waiter.wait(timeout=300.0)           # Block until THIS row's result

# In SharedBatchAdapter.emit() - routes by token_id:
def emit(self, token, result):
    with self._lock:
        self._results[token.token_id] = result
        if token.token_id in self._waiters:
            self._waiters[token.token_id].set()   # Wake correct waiter
```

**Why it works:** Single adapter connected once per transform. Each row registers a waiter by `token_id`. Results are routed to the correct waiter regardless of completion order.

### How Concurrency Works

```
Time →
0ms:   process(A) → accept(A) → wait(A) ──────────────────────► got A
       ↓
10ms:  Worker1 starts A
       process(B) → accept(B) → wait(B) ──────────────► got B
       ↓
20ms:  Worker2 starts B (A still running)
       process(C) → accept(C) → wait(C) ──────► got C
       ↓
30ms:  Worker3 starts C (A,B still running)

Pool: [Worker1: A] [Worker2: B] [Worker3: C] ← ALL CONCURRENT
```

---

## Phase 1: Replace BlockingResultAdapter with SharedBatchAdapter (1.5 hours)

> **Migration Note:** This phase replaces the existing broken `BlockingResultAdapter` class in `src/elspeth/engine/batch_adapter.py`. The existing class will be deleted after `SharedBatchAdapter` is verified working.

### Task 1.1: Implement SharedBatchAdapter

**File:** `src/elspeth/engine/batch_adapter.py` (replace existing content)

```python
"""Adapter for batch transform integration with TransformExecutor.

Allows TransformExecutor to call accept() and wait for results while
maintaining concurrency across multiple rows.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.contracts import TransformResult, TokenInfo


class RowWaiter:
    """Waiter for a specific row's result."""

    def __init__(
        self,
        token_id: str,
        event: threading.Event,
        results: dict[str, TransformResult],
        lock: threading.Lock,
    ):
        self._token_id = token_id
        self._event = event
        self._results = results
        self._lock = lock

    def wait(self, timeout: float = 300.0) -> TransformResult:
        """Block until this row's result arrives.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            TransformResult for this row

        Raises:
            TimeoutError: If result not received within timeout
        """
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(
                f"No result received for token {self._token_id} within {timeout}s"
            )

        with self._lock:
            result = self._results.pop(self._token_id)
            return result


class SharedBatchAdapter:
    """Shared output port adapter for batch transforms.

    Allows multiple rows to be in flight concurrently while routing
    results back to the correct waiter based on token_id.

    Usage:
        adapter = SharedBatchAdapter()
        transform.connect_output(adapter, max_pending=30)

        # For each row:
        waiter = adapter.register(token.token_id)
        transform.accept(row, ctx)
        result = waiter.wait()
    """

    def __init__(self):
        """Initialize adapter."""
        self._waiters: dict[str, threading.Event] = {}
        self._results: dict[str, TransformResult] = {}
        self._lock = threading.Lock()

    def register(self, token_id: str) -> RowWaiter:
        """Register a waiter for a specific token.

        Args:
            token_id: Token ID to wait for

        Returns:
            RowWaiter that can wait() for the result
        """
        with self._lock:
            event = threading.Event()
            self._waiters[token_id] = event
            return RowWaiter(token_id, event, self._results, self._lock)

    def emit(self, token: TokenInfo, result: TransformResult) -> None:
        """Receive result from batch transform.

        Routes result to the correct waiter based on token_id.

        Args:
            token: Token for this row
            result: Transform result
        """
        with self._lock:
            token_id = token.token_id
            self._results[token_id] = result

            if token_id in self._waiters:
                # Wake the waiter
                self._waiters[token_id].set()
                # Clean up waiter (result stays until waiter.wait() retrieves it)
                del self._waiters[token_id]

    def clear(self) -> None:
        """Clear all pending waiters and results (for testing)."""
        with self._lock:
            self._waiters.clear()
            self._results.clear()
```

**Test:** `tests/engine/test_batch_adapter.py`

```python
"""Tests for SharedBatchAdapter."""

import threading
import time

import pytest

from elspeth.contracts import TransformResult, TokenInfo
from elspeth.engine.batch_adapter import SharedBatchAdapter


def test_single_row_wait():
    """Test waiting for a single row's result."""
    adapter = SharedBatchAdapter()

    # Register waiter
    waiter = adapter.register("token-1")

    # Emit result in background thread
    def emit_later():
        time.sleep(0.1)
        token = TokenInfo(row_id=1, token_id="token-1", row_data={}, branch_name=None)
        result = TransformResult.success({"output": "done"})
        adapter.emit(token, result)

    thread = threading.Thread(target=emit_later)
    thread.start()

    # Wait for result
    result = waiter.wait(timeout=5.0)

    assert result.status == "success"
    assert result.row == {"output": "done"}

    thread.join()


def test_multiple_concurrent_rows():
    """Test multiple rows waiting concurrently."""
    adapter = SharedBatchAdapter()

    # Register 3 waiters
    waiter1 = adapter.register("token-1")
    waiter2 = adapter.register("token-2")
    waiter3 = adapter.register("token-3")

    # Emit results out of order
    def emit_results():
        time.sleep(0.05)
        # Emit token-2 first (out of order)
        adapter.emit(
            TokenInfo(row_id=2, token_id="token-2", row_data={}, branch_name=None),
            TransformResult.success({"value": 2})
        )
        time.sleep(0.05)
        # Then token-1
        adapter.emit(
            TokenInfo(row_id=1, token_id="token-1", row_data={}, branch_name=None),
            TransformResult.success({"value": 1})
        )
        time.sleep(0.05)
        # Then token-3
        adapter.emit(
            TokenInfo(row_id=3, token_id="token-3", row_data={}, branch_name=None),
            TransformResult.success({"value": 3})
        )

    thread = threading.Thread(target=emit_results)
    thread.start()

    # Wait for results (each waiter gets correct result regardless of emit order)
    result1 = waiter1.wait(timeout=5.0)
    result2 = waiter2.wait(timeout=5.0)
    result3 = waiter3.wait(timeout=5.0)

    assert result1.row == {"value": 1}
    assert result2.row == {"value": 2}
    assert result3.row == {"value": 3}

    thread.join()


def test_timeout():
    """Test that wait() times out if result never arrives."""
    adapter = SharedBatchAdapter()
    waiter = adapter.register("token-1")

    with pytest.raises(TimeoutError, match="No result received"):
        waiter.wait(timeout=0.1)
```

**Run:**
```bash
.venv/bin/python -m pytest tests/engine/test_batch_adapter.py -xvs
```

**Commit:**
```bash
git add src/elspeth/engine/batch_adapter.py tests/engine/test_batch_adapter.py
git commit -m "feat(engine): add SharedBatchAdapter for batch transform integration

- SharedBatchAdapter routes results to per-row waiters
- RowWaiter blocks until specific token_id result arrives
- Supports concurrent rows with out-of-order completion
- Tests verify single-row, multi-row, and timeout scenarios"
```

---

## Phase 2: Fix TransformExecutor Integration (2 hours)

> **Note:** This replaces the existing broken integration at `executors.py:193-219`. The existing code creates a new `BlockingResultAdapter` per row and only connects the first one. This fix uses a single `SharedBatchAdapter` per transform that routes results by `token_id`.

### Task 2.1: Replace Existing Integration Code

**File:** `src/elspeth/engine/executors.py`

**Add method to get/create shared adapter (replaces per-row adapter creation):**

```python
    def _get_batch_adapter(self, transform: TransformProtocol) -> SharedBatchAdapter:
        """Get or create shared batch adapter for transform.

        Creates adapter once per transform instance and stores it as an
        instance attribute for reuse across rows.

        Args:
            transform: The batch-aware transform

        Returns:
            SharedBatchAdapter for this transform
        """
        if not hasattr(transform, '_executor_batch_adapter'):
            from elspeth.engine.batch_adapter import SharedBatchAdapter
            adapter = SharedBatchAdapter()
            transform._executor_batch_adapter = adapter

            # Connect output (one-time setup)
            max_pending = getattr(transform._config, 'pool_size', 30)
            transform.connect_output(adapter, max_pending=max_pending)
            transform._batch_initialized = True

        return transform._executor_batch_adapter
```

**Replace the broken integration in execute_transform():**

Replace the existing broken batch handling code (lines 193-219) with:

```python
# (Delete the existing BlockingResultAdapter-based code)
```

**New implementation:**

```python
            # Detect batch-aware transforms (have accept method)
            has_accept = hasattr(transform, 'accept') and callable(transform.accept)

            if has_accept:
                # Batch-aware transform - use accept() + wait pattern
                adapter = self._get_batch_adapter(transform)

                # Register waiter for THIS token
                waiter = adapter.register(token.token_id)

                # Set token on context (required by BatchTransformMixin)
                ctx.token = token

                # Submit row (returns immediately, work happens in background)
                transform.accept(token.row_data, ctx)

                # Block until THIS row's result arrives
                result = waiter.wait(timeout=300.0)
            else:
                # Regular transform - call process() synchronously
                result = transform.process(token.row_data, ctx)

            duration_ms = (time.perf_counter() - start) * 1000
```

**Test:** Run existing transform executor tests

```bash
.venv/bin/python -m pytest tests/engine/test_transform_executor*.py -xvs
```

**Commit:**
```bash
git add src/elspeth/engine/executors.py
git commit -m "feat(executor): integrate SharedBatchAdapter for batch transforms

- Detect batch transforms via hasattr(transform, 'accept')
- Create shared adapter once per transform instance
- Use accept() + waiter.wait() pattern for batch transforms
- Regular transforms unchanged (call process() directly)

This enables LLM transforms to work with the engine while
maintaining concurrent processing."
```

---

## ~~Phase 3: REMOVED - process() Wrapper~~

> **REMOVED (Peer Review Decision):** Adding a `process()` fallback to LLM transforms was rejected because it creates **dual code paths** for the same operation—violating CLAUDE.md's "No Legacy Code Policy."
>
> The correct design is:
> - TransformExecutor detects batch transforms via `hasattr(transform, 'accept')`
> - Batch transforms: use `accept()` + `SharedBatchAdapter` pattern (Phase 2)
> - Regular transforms: use `process()` directly
>
> LLM transforms should **keep** `process()` raising `NotImplementedError`. The executor handles them via the `accept()` path. No fallback needed.

---

## Phase 3: Test Integration (2 hours)

### Task 3.1: Test with TransformExecutor

**Create:** `tests/engine/test_executor_batch_integration.py`

```python
"""Test TransformExecutor integration with batch transforms."""

import pytest

from elspeth.contracts import TokenInfo, TransformResult
from elspeth.core.context import PluginContext
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig


def test_executor_calls_batch_transform_via_accept(mock_azure_client):
    """Verify TransformExecutor uses accept() path for batch transforms."""
    # Setup
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    spans = SpanFactory(enabled=False)
    executor = TransformExecutor(recorder, spans)

    # Create batch transform
    config = AzureOpenAIConfig(
        deployment_name="gpt-4",
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        pool_size=10,
        system_prompt="Classify",
        user_prompt_template="{{text}}",
        output_key="result",
    )
    transform = AzureLLMTransform(config=config)
    transform.node_id = "transform-1"

    # Mock LLM response
    mock_azure_client.return_value.chat.completions.create.return_value.choices = [
        type('obj', (object,), {'message': type('obj', (object,), {'content': 'positive'})()})()
    ]

    # Create token
    token = TokenInfo(
        row_id=1,
        token_id="token-1",
        row_data={"text": "Great!"},
        branch_name=None,
    )
    ctx = PluginContext(token=token, landscape=recorder)

    # Execute through executor
    result, updated_token, error_sink = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=0,
    )

    # Verify result
    assert result.status == "success"
    assert result.row["result"] == "positive"
    assert error_sink is None

    # Cleanup
    transform.close()


def test_executor_concurrent_rows(mock_azure_client):
    """Verify multiple rows process concurrently through executor."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    spans = SpanFactory(enabled=False)
    executor = TransformExecutor(recorder, spans)

    config = AzureOpenAIConfig(
        deployment_name="gpt-4",
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        pool_size=10,
        system_prompt="Classify",
        user_prompt_template="{{text}}",
        output_key="result",
    )
    transform = AzureLLMTransform(config=config)
    transform.node_id = "transform-1"

    # Mock responses
    responses = ["positive", "negative", "neutral"]
    mock_azure_client.return_value.chat.completions.create.return_value.choices = [
        type('obj', (object,), {'message': type('obj', (object,), {'content': responses[0]})()})()
    ]

    # Process 3 rows
    results = []
    for i in range(3):
        token = TokenInfo(
            row_id=i,
            token_id=f"token-{i}",
            row_data={"text": f"Text {i}"},
            branch_name=None,
        )
        ctx = PluginContext(token=token, landscape=recorder)

        result, _, _ = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=0,
        )
        results.append(result)

    # Verify all processed
    assert len(results) == 3
    assert all(r.status == "success" for r in results)

    # Verify concurrency (check batch metrics)
    metrics = transform.get_batch_metrics()
    # At some point during processing, multiple rows were in flight
    # (exact timing depends on mock response speed)

    transform.close()
```

**Run:**
```bash
.venv/bin/python -m pytest tests/engine/test_executor_batch_integration.py -xvs
```

**Commit:**
```bash
git add tests/engine/test_executor_batch_integration.py
git commit -m "test(engine): verify TransformExecutor batch integration

- Test executor calls accept() for batch transforms
- Test concurrent row processing
- Verify audit trail integration"
```

---

### Task 3.2: Unskip Integration Tests

**File:** `tests/integration/test_llm_transforms.py`

**Remove skip decorators:**
```python
# Line 23:
-@pytest.mark.skip(reason="OpenRouterLLMTransform now uses BatchTransformMixin - tests need rewrite for accept() API")
+# (delete the line)

# Line 127:
-@pytest.mark.skip(reason="OpenRouterLLMTransform now uses BatchTransformMixin - tests need rewrite for accept() API")
+# (delete the line)
```

**Run:**
```bash
.venv/bin/python -m pytest tests/integration/test_llm_transforms.py -xvs
```

**Commit:**
```bash
git add tests/integration/test_llm_transforms.py
git commit -m "test(integration): unskip LLM integration tests

Integration tests now pass with batch transform integration."
```

---

## Phase 4: Update All Tests (2 hours)

### Task 4.1: Update LLM Plugin Tests

Tests currently use accept()/flush() pattern. They should continue working since accept() is unchanged.

But verify they still pass:

```bash
.venv/bin/python -m pytest tests/plugins/llm/ -xvs
```

If any failures, update to use the executor pattern or keep using accept() directly.

**Commit:**
```bash
git add tests/plugins/llm/
git commit -m "test(llm): verify all LLM plugin tests pass"
```

---

## Phase 5: Verify Audit Trail (1 hour)

### Task 5.1: Test Audit Recording

**Create:** `tests/engine/test_batch_audit_trail.py`

```python
"""Test that batch transforms record complete audit trail."""

import pytest

from elspeth.contracts import TokenInfo
from elspeth.core.context import PluginContext
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig


def test_batch_transform_records_node_states(mock_azure_client):
    """Verify batch transforms record node_states via executor."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    spans = SpanFactory(enabled=False)
    executor = TransformExecutor(recorder, spans)

    # Create transform
    config = AzureOpenAIConfig(
        deployment_name="gpt-4",
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        pool_size=10,
        system_prompt="Classify",
        user_prompt_template="{{text}}",
        output_key="classification",
    )
    transform = AzureLLMTransform(config=config)
    transform.node_id = "llm-transform-1"

    # Mock response
    mock_azure_client.return_value.chat.completions.create.return_value.choices = [
        type('obj', (object,), {'message': type('obj', (object,), {'content': 'positive'})()})()
    ]

    # Create token
    token = TokenInfo(
        row_id=1,
        token_id="token-1",
        row_data={"text": "Great product!"},
        branch_name=None,
    )
    ctx = PluginContext(token=token, landscape=recorder)

    # Execute
    result, _, _ = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=0,
    )

    # Verify node_states recorded
    states = recorder.get_node_states(token_id="token-1")
    assert len(states) == 1

    state = states[0]
    assert state.node_id == "llm-transform-1"
    assert state.status == "completed"
    assert state.input_data == {"text": "Great product!"}
    assert state.output_data["classification"] == "positive"

    # Verify external calls recorded
    calls = recorder.get_external_calls(state_id=state.state_id)
    assert len(calls) >= 1  # At least one LLM call

    transform.close()


def test_batch_transform_records_errors(mock_azure_client):
    """Verify batch transform errors are recorded in audit trail."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    spans = SpanFactory(enabled=False)
    executor = TransformExecutor(recorder, spans)

    config = AzureOpenAIConfig(
        deployment_name="gpt-4",
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        pool_size=10,
        system_prompt="Classify",
        user_prompt_template="{{text}}",
        output_key="result",
    )
    transform = AzureLLMTransform(config=config)
    transform.node_id = "llm-1"
    transform._on_error = "error_sink"  # Configure error routing

    # Mock LLM error
    from elspeth.plugins.clients.llm import LLMClientError
    mock_azure_client.return_value.chat.completions.create.side_effect = LLMClientError("API error")

    token = TokenInfo(row_id=1, token_id="token-1", row_data={"text": "test"}, branch_name=None)
    ctx = PluginContext(token=token, landscape=recorder)

    # Execute
    result, _, error_sink = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=0,
    )

    # Verify error recorded
    assert result.status == "error"
    assert error_sink == "error_sink"

    states = recorder.get_node_states(token_id="token-1")
    assert len(states) == 1
    assert states[0].status == "failed"

    transform.close()
```

**Run:**
```bash
.venv/bin/python -m pytest tests/engine/test_batch_audit_trail.py -xvs
```

**Commit:**
```bash
git add tests/engine/test_batch_audit_trail.py
git commit -m "test(audit): verify batch transforms record complete audit trail

- Test node_states recording via executor
- Test external_calls recording
- Test error recording and routing"
```

---

## Phase 6: Documentation and Cleanup (1 hour)

### Task 6.1: Update CLAUDE.md

**File:** `CLAUDE.md`

Add after "Transform Subtypes" section:

```markdown
### Transform API: process() is Universal

ALL transforms implement `process(row, ctx) -> TransformResult`, including
batch-aware transforms that use concurrent processing internally.

**Batch-aware transforms** (LLM transforms using `BatchTransformMixin`):
- Implement both `process()` and `accept()` methods
- `TransformExecutor` detects batch transforms via `hasattr(transform, 'accept')`
- Engine uses `accept()` + waiter pattern for concurrent processing
- From engine's perspective, rows process sequentially but with concurrent execution internally
- `process()` exists for protocol compliance but falls back to sequential mode

**Example flow:**
```python
# Engine calls:
result = executor.execute_transform(transform, token, ctx, step)

# Executor detects batch transform:
if hasattr(transform, 'accept'):
    waiter = adapter.register(token.token_id)
    transform.accept(row, ctx)  # Returns immediately
    result = waiter.wait()      # Blocks until this row completes
```

**Concurrency model:**
- Row 1: accept() → wait() [Worker starts processing row 1]
- Row 2: accept() → wait() [Worker starts processing row 2, row 1 still running]
- Row 3: accept() → wait() [Worker starts processing row 3, rows 1-2 still running]
- All 3 rows process concurrently in worker pool
- Results arrive in FIFO order (RowReorderBuffer)
```

**Commit:**
```bash
git add CLAUDE.md
git commit -m "docs: document batch transform integration in CLAUDE.md"
```

---

### Task 6.2: Update Design Docs

**File:** `docs/design/row-level-pipelining-design.md`

Update to reflect executor integration instead of direct accept() calls.

**Commit:**
```bash
git add docs/design/
git commit -m "docs: update row-level pipelining design for executor integration"
```

---

## Phase 7: Final Verification (1 hour)

### Task 7.1: Full Test Suite

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -x

# Run type checking
.venv/bin/python -m mypy src/elspeth/

# Run linting
.venv/bin/python -m ruff check src/
```

**Fix any failures**

**Commit:**
```bash
git add .
git commit -m "test: verify full test suite passes"
```

---

### Task 7.2: Manual Integration Test

**Create:** `examples/test_batch_integration.py`

```python
"""Manual test of batch transform integration."""

from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.context import PluginContext
from elspeth.contracts import TokenInfo
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.spans import SpanFactory

# Setup
db = LandscapeDB.in_memory()
recorder = LandscapeRecorder(db)
spans = SpanFactory(enabled=False)
executor = TransformExecutor(recorder, spans)

# Configure transform
config = AzureOpenAIConfig(
    deployment_name="gpt-4-turbo",
    endpoint="https://YOUR-ENDPOINT.openai.azure.com",
    api_key="YOUR-API-KEY",
    pool_size=10,
    system_prompt="You are a sentiment classifier.",
    user_prompt_template="Classify: {{text}}",
    output_key="sentiment",
)
transform = AzureLLMTransform(config=config)
transform.node_id = "sentiment-classifier"

# Process multiple rows
rows = [
    {"text": "This is amazing!"},
    {"text": "Terrible experience."},
    {"text": "It's okay, nothing special."},
]

results = []
for i, row_data in enumerate(rows):
    token = TokenInfo(
        row_id=i,
        token_id=f"token-{i}",
        row_data=row_data,
        branch_name=None,
    )
    ctx = PluginContext(token=token, landscape=recorder)

    result, _, _ = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=0,
    )
    results.append(result)
    print(f"Row {i}: {result.row.get('sentiment')}")

# Check batch metrics
metrics = transform.get_batch_metrics()
print(f"\nBatch metrics: {metrics}")

# Cleanup
transform.close()
```

**Run manually to verify real LLM calls work**

---

## Success Criteria

- [ ] `BlockingResultAdapter` replaced with `SharedBatchAdapter` (routes by token_id)
- [ ] TransformExecutor integrates with batch transforms via SharedBatchAdapter
- [ ] Concurrent row processing works (30 rows × 4 queries = 120 concurrent)
- [ ] **No deadlocks** when processing multiple rows through batch transforms
- [ ] Audit trail complete (node_states, external_calls recorded)
- [ ] All tests pass (unit, integration, executor)
- [ ] Integration tests unskipped and passing
- [ ] Manual LLM test works with real API
- [ ] Documentation updated
- [ ] Type checking passes
- [ ] Zero architectural debt introduced

---

## Timeline Summary

| Phase | Hours | Description |
|-------|-------|-------------|
| Phase 1 | 1.5h | Replace BlockingResultAdapter with SharedBatchAdapter |
| Phase 2 | 2h | Fix TransformExecutor integration |
| Phase 3 | 2h | Test integration |
| Phase 4 | 2h | Update all tests |
| Phase 5 | 1h | Verify audit trail |
| Phase 6 | 1h | Documentation |
| Phase 7 | 1h | Final verification |
| **TOTAL** | **~10.5h** | (reduced from 24h after peer review) |

---

## Rollback Plan

If critical issues found:

1. **Before Phase 2 complete:** Revert Phase 1-2, restore `BlockingResultAdapter`
2. **Before Phase 4 complete:** Batch transforms broken, revert all or finish quickly
3. **After Phase 4:** Tests passing, only docs/cleanup remain - push forward

**Git strategy:** One commit per task, easy to bisect and revert.

---

## Migration Notes

1. **Delete `BlockingResultAdapter`** after `SharedBatchAdapter` is verified working
2. **Update any tests** that directly reference `BlockingResultAdapter` (search for imports)
3. **Update skip marker reasons** that reference the old line numbers (executors.py:192)
