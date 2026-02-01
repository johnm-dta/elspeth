# Unify Transform API to `process()` - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `process()` the universal transform interface by having batch transforms return futures that block until row completion.

**Architecture:** Batch transforms implement `process()` by submitting rows to internal thread pool infrastructure and blocking on a future until that specific row completes. The engine sees synchronous `process()` calls; concurrency happens inside the batch infrastructure.

**Tech Stack:** Python 3.11+, `concurrent.futures.Future`, `threading`, `RowReorderBuffer` (existing)

**Critical Decisions (from peer review):**
1. **Output ports eliminated** - Batch transforms return `TransformResult` from `process()`, no longer emit to ports
2. **Context threading** - Each row gets its own `PluginContext` instance passed to worker thread
3. **FIFO ordering preserved** - Existing `RowReorderBuffer` reordering logic ensures FIFO
4. **Timeout from config** - Use existing `_processing_timeout` config field

---

## Phase 0: Preparation and Discovery

### Task 0.1: Verify FIFO Reordering Logic

**Files:**
- Read: `src/elspeth/plugins/batching/row_reorder_buffer.py`
- Read: `src/elspeth/plugins/batching/mixin.py:205-229` (_release_loop)

**Step 1: Read the RowReorderBuffer implementation**

Read: `src/elspeth/plugins/batching/row_reorder_buffer.py`

Expected: Find `wait_for_next_release()` method that blocks until the next FIFO-ordered entry is ready.

**Step 2: Verify _release_loop maintains FIFO order**

Read: `src/elspeth/plugins/batching/mixin.py` lines 205-229

Expected: `_release_loop` calls `wait_for_next_release()` which guarantees FIFO ordering.

**Step 3: Document findings**

Write findings to `docs/plans/2026-01-26-fifo-verification.md`:
```markdown
# FIFO Ordering Verification

**Status:** ✅ Verified

`RowReorderBuffer.wait_for_next_release()` maintains FIFO ordering by:
- Tracking submission sequence number
- Blocking until next expected sequence number is complete
- Buffering out-of-order completions

This means futures can resolve out-of-order internally, but we can still
guarantee FIFO if needed (though for `process()` return values, we don't
need FIFO enforcement since each caller blocks on their own future).
```

**Step 4: Commit**

```bash
git add docs/plans/2026-01-26-fifo-verification.md
git commit -m "docs: verify FIFO ordering in RowReorderBuffer"
```

---

## Phase 1: Add Future-Based API to BatchTransformMixin

### Task 1.1: Add Future Tracking Infrastructure

**Files:**
- Modify: `src/elspeth/plugins/batching/mixin.py`
- Test: `tests/plugins/batching/test_batch_transform_mixin.py`

**Step 1: Write test for future-based submission**

Add to `tests/plugins/batching/test_batch_transform_mixin.py`:

```python
def test_submit_row_returns_future():
    """Test that _submit_row returns a future that resolves to TransformResult."""
    from concurrent.futures import Future
    from elspeth.plugins.batching import BatchTransformMixin
    from elspeth.contracts import TransformResult
    from elspeth.core.context import PluginContext
    from elspeth.core.token import TokenInfo

    # Create a test transform that uses the mixin
    class TestTransform(BatchTransformMixin):
        def __init__(self):
            self._futures = {}
            self._batch_buffer = None
            self._batch_executor = None

        def _process_row(self, row, ctx):
            return TransformResult.success({"result": row["x"] * 2})

    transform = TestTransform()
    # Initialize batch infrastructure (we'll need to add this method)
    transform.init_batch_processing_with_futures(max_pending=10, name="test")

    # Create context with token
    token = TokenInfo(row_id=1, token_id=1, row_data={"x": 5}, branch_name=None)
    ctx = PluginContext(token=token, landscape=None)

    # Submit row and get future
    future = transform._submit_row({"x": 5}, ctx)

    # Verify it's a Future
    assert isinstance(future, Future)

    # Block on result
    result = future.result(timeout=5.0)

    # Verify result
    assert result.status == "success"
    assert result.row == {"result": 10}

    # Cleanup
    transform.shutdown_batch_processing()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plugins/batching/test_batch_transform_mixin.py::test_submit_row_returns_future -xvs`

Expected: `AttributeError: 'TestTransform' object has no attribute '_submit_row'`

**Step 3: Add future tracking to BatchTransformMixin**

Modify `src/elspeth/plugins/batching/mixin.py`:

```python
from concurrent.futures import Future
from typing import Dict

class BatchTransformMixin:
    """Mixin that adds concurrent row processing to any transform.

    ... (existing docstring) ...
    """

    # Add new attributes for future tracking
    _batch_futures: Dict[int, Future]  # Maps sequence_num -> Future
    _batch_sequence_lock: threading.Lock
    _batch_next_sequence: int
```

Add after line 106 in `init_batch_processing()`:

```python
        # Future tracking for process() API
        self._batch_futures = {}
        self._batch_sequence_lock = threading.Lock()
        self._batch_next_sequence = 0
```

**Step 4: Add _submit_row method**

Add new method to `BatchTransformMixin` after `accept_row()`:

```python
    def _submit_row(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
        processor: Callable[[dict[str, Any], PluginContext], TransformResult],
    ) -> Future[TransformResult]:
        """Submit a row for processing and return a future.

        This is the future-based API for process() implementations.
        Unlike accept_row() which emits to output port, this returns
        a future that resolves when the specific row completes.

        Args:
            row: The row data
            ctx: Plugin context (must have ctx.token set)
            processor: Function that does actual processing

        Returns:
            Future that resolves to TransformResult when row completes

        Raises:
            ValueError: If ctx.token is None
            ShutdownError: If batch processing is shut down
        """
        # No defensive fallback - ctx.token is required
        if ctx.token is None:
            raise ValueError(
                "BatchTransformMixin requires ctx.token to be set. "
                "This is a bug in the calling code."
            )

        token = ctx.token

        # Allocate sequence number under lock
        with self._batch_sequence_lock:
            sequence_num = self._batch_next_sequence
            self._batch_next_sequence += 1

            # Create future for this row
            future: Future[TransformResult] = Future()
            self._batch_futures[sequence_num] = future

        # Submit to buffer (blocks on backpressure)
        ticket = self._batch_buffer.submit(sequence_num)

        # Submit to worker pool
        self._batch_executor.submit(
            self._process_and_complete_with_future,
            ticket,
            sequence_num,
            token,
            row,
            ctx,
            processor,
        )

        return future
```

**Step 5: Add worker method that resolves futures**

Add new method after `_process_and_complete()`:

```python
    def _process_and_complete_with_future(
        self,
        ticket: RowTicket,
        sequence_num: int,
        token: TokenInfo,
        row: dict[str, Any],
        ctx: PluginContext,
        processor: Callable[[dict[str, Any], PluginContext], TransformResult],
    ) -> None:
        """Worker thread: process row, resolve future, mark complete.

        Called by worker threads for the future-based API. Executes the
        processor function and resolves the future with the result or exception.
        """
        future = self._batch_futures.get(sequence_num)
        if future is None:
            # This is a bug - future should have been created
            raise RuntimeError(
                f"Future for sequence {sequence_num} not found. This is a bug."
            )

        try:
            result = processor(row, ctx)
            # Resolve future with success
            future.set_result(result)
        except Exception as e:
            # Propagate exception through future
            future.set_exception(e)

            # Also create error result for buffer completion
            tb = traceback.format_exc()
            from elspeth.contracts import TransformResult
            result = TransformResult.error(
                {
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback": tb,
                }
            )
        finally:
            # Mark complete in buffer (for metrics/shutdown)
            self._batch_buffer.complete(ticket, (token, result))

            # Clean up future reference
            with self._batch_sequence_lock:
                self._batch_futures.pop(sequence_num, None)
```

**Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/plugins/batching/test_batch_transform_mixin.py::test_submit_row_returns_future -xvs`

Expected: Test passes

**Step 7: Commit**

```bash
git add src/elspeth/plugins/batching/mixin.py tests/plugins/batching/test_batch_transform_mixin.py
git commit -m "feat(batching): add future-based _submit_row API

- Add _batch_futures dict to track pending futures
- Add _submit_row() that returns Future[TransformResult]
- Add _process_and_complete_with_future() worker
- Exceptions propagate through future.set_exception()
- Future cleaned up after resolution

This enables process() implementations that block on futures while
maintaining concurrent execution in the worker pool."
```

---

### Task 1.2: Test Exception Propagation Through Futures

**Files:**
- Test: `tests/plugins/batching/test_batch_transform_mixin.py`

**Step 1: Write test for exception propagation**

Add to `tests/plugins/batching/test_batch_transform_mixin.py`:

```python
def test_submit_row_propagates_exceptions():
    """Test that exceptions in processor are propagated through the future."""
    from concurrent.futures import Future
    from elspeth.plugins.batching import BatchTransformMixin
    from elspeth.core.context import PluginContext
    from elspeth.core.token import TokenInfo

    class TestTransform(BatchTransformMixin):
        def __init__(self):
            self._futures = {}

        def _process_row(self, row, ctx):
            raise ValueError("Intentional test error")

    transform = TestTransform()
    transform.init_batch_processing_with_futures(max_pending=10, name="test")

    token = TokenInfo(row_id=1, token_id=1, row_data={"x": 5}, branch_name=None)
    ctx = PluginContext(token=token, landscape=None)

    # Submit row
    future = transform._submit_row({"x": 5}, ctx)

    # Verify exception is raised when we call result()
    import pytest
    with pytest.raises(ValueError, match="Intentional test error"):
        future.result(timeout=5.0)

    # Cleanup
    transform.shutdown_batch_processing()
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/plugins/batching/test_batch_transform_mixin.py::test_submit_row_propagates_exceptions -xvs`

Expected: Test passes (implementation from previous task handles this)

**Step 3: Commit**

```bash
git add tests/plugins/batching/test_batch_transform_mixin.py
git commit -m "test(batching): verify exception propagation through futures"
```

---

### Task 1.3: Add Timeout Test

**Files:**
- Test: `tests/plugins/batching/test_batch_transform_mixin.py`

**Step 1: Write test for timeout behavior**

Add to `tests/plugins/batching/test_batch_transform_mixin.py`:

```python
def test_submit_row_timeout():
    """Test that future.result() respects timeout."""
    import time
    from concurrent.futures import TimeoutError
    from elspeth.plugins.batching import BatchTransformMixin
    from elspeth.contracts import TransformResult
    from elspeth.core.context import PluginContext
    from elspeth.core.token import TokenInfo

    class SlowTransform(BatchTransformMixin):
        def __init__(self):
            self._futures = {}

        def _process_row(self, row, ctx):
            # Sleep longer than timeout
            time.sleep(2.0)
            return TransformResult.success({"done": True})

    transform = SlowTransform()
    transform.init_batch_processing_with_futures(max_pending=10, name="slow")

    token = TokenInfo(row_id=1, token_id=1, row_data={"x": 5}, branch_name=None)
    ctx = PluginContext(token=token, landscape=None)

    # Submit row
    future = transform._submit_row({"x": 5}, ctx)

    # Verify timeout is raised
    import pytest
    with pytest.raises(TimeoutError):
        future.result(timeout=0.1)  # Very short timeout

    # Cleanup (wait for slow work to finish)
    transform.shutdown_batch_processing(timeout=5.0)
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/plugins/batching/test_batch_transform_mixin.py::test_submit_row_timeout -xvs`

Expected: Test passes

**Step 3: Commit**

```bash
git add tests/plugins/batching/test_batch_transform_mixin.py
git commit -m "test(batching): verify future timeout behavior"
```

---

## Phase 2: Update LLM Transforms to Use process()

### Task 2.1: Implement process() in AzureLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/azure.py`
- Test: `tests/plugins/llm/test_azure.py`

**Step 1: Write test for process() API**

Add to `tests/plugins/llm/test_azure.py`:

```python
def test_azure_llm_process_api(mock_azure_client, test_landscape):
    """Test that AzureLLMTransform.process() works correctly."""
    from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig
    from elspeth.core.context import PluginContext
    from elspeth.core.token import TokenInfo

    # Configure transform
    config = AzureOpenAIConfig(
        deployment_name="gpt-4",
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        pool_size=2,
        system_prompt="You are a classifier",
        user_prompt_template="Classify: {{text}}",
        output_key="classification",
    )

    transform = AzureLLMTransform(config=config)

    # Mock LLM response
    mock_azure_client.return_value.chat.completions.create.return_value = MockResponse(
        content="positive"
    )

    # Create context with token
    token = TokenInfo(
        row_id=1,
        token_id=1,
        row_data={"text": "This is great!"},
        branch_name=None,
    )
    ctx = PluginContext(token=token, landscape=test_landscape, state_id=1)

    # Call process() directly (new API)
    result = transform.process({"text": "This is great!"}, ctx)

    # Verify result
    assert result.status == "success"
    assert result.row["classification"] == "positive"
    assert "text" in result.row  # Original field preserved

    # Cleanup
    transform.close()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure.py::test_azure_llm_process_api -xvs`

Expected: `NotImplementedError: AzureLLMTransform uses row-level pipelining...`

**Step 3: Replace process() implementation in AzureLLMTransform**

Modify `src/elspeth/plugins/llm/azure.py` lines 225-242:

```python
    def process(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row through Azure OpenAI.

        Submits the row to the batch processing infrastructure and blocks
        until the result is ready. While blocking, other rows can be
        submitted and processed concurrently by the worker pool.

        Args:
            row: Row to process
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult with classification/generation added to row

        Raises:
            TimeoutError: If processing exceeds timeout
            LLMClientError: If LLM call fails
        """
        # Submit to batch infrastructure and get future
        future = self._submit_row(row, ctx, self._process_row)

        # Block until THIS row completes
        # Use configured timeout (default 120s from LLMConfig)
        timeout = getattr(self._config, 'timeout', 120.0)
        return future.result(timeout=timeout)
```

**Step 4: Initialize batch processing with futures in __init__**

Find the `__init__` method and replace the `init_batch_processing()` call:

Before (around line 150-160):
```python
        self.init_batch_processing(
            max_pending=config.pool_size,
            output=output,
            name=f"azure-llm-{config.deployment_name}",
        )
```

After:
```python
        # Initialize batch processing with future-based API
        # No output port needed - process() returns results directly
        self._init_batch_with_futures(
            max_pending=config.pool_size,
            name=f"azure-llm-{config.deployment_name}",
        )
```

Wait - we need to update `BatchTransformMixin` to support initialization without output port. Let me revise this step.

**Step 4 (revised): Update init to not require output port**

Actually, looking at the code, we should keep the existing `init_batch_processing()` but add a flag to indicate we're using the future-based API. Let's modify the approach:

Modify `src/elspeth/plugins/llm/azure.py` around line 150:

```python
    def __init__(self, config: AzureOpenAIConfig) -> None:
        """Initialize the Azure LLM transform.

        Args:
            config: Azure OpenAI configuration
        """
        super().__init__(config)
        self._config = config
        self._batch_initialized = False

        # Will be set when batch processing initializes
        self._recorder: LandscapeRecorder | None = None

        # Create prompt template
        self._prompt_template = PromptTemplate(
            system_prompt=config.system_prompt,
            user_prompt_template=config.user_prompt_template,
        )

        # Output schema (if json_output enabled)
        self._output_schema = None
        if config.json_output:
            self._output_schema = create_schema_from_config(
                config.output_schema,
                schema_name=f"{config.deployment_name}_output",
            )

        # LLM client initialization is lazy (happens in _get_client)
        self._client: AzureOpenAI | None = None
        self._client_lock = Lock()

        # Batch processing initialization
        # For process() API, we don't need an output port
        # We'll use a dummy port for now to satisfy init_batch_processing
        from elspeth.plugins.batching.ports import CollectorOutputPort
        dummy_port = CollectorOutputPort()

        self.init_batch_processing(
            max_pending=config.pool_size,
            output=dummy_port,
            name=f"azure-llm-{config.deployment_name}",
        )
        self._batch_initialized = True
```

Actually, this is getting messy. Let me think about this differently. Looking at the peer review, the recommendation was to eliminate output ports entirely for the future-based API. Let me add a new initialization method to `BatchTransformMixin` specifically for the future-based API.

Let me revise the plan: we should add `init_batch_processing_futures()` to `BatchTransformMixin` that doesn't require an output port.

**Step 4 (re-revised): Add to BatchTransformMixin first**

Go back to `src/elspeth/plugins/batching/mixin.py` and add:

```python
    def init_batch_processing_futures(
        self,
        max_pending: int,
        name: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        """Initialize batch processing for future-based API (no output port).

        Use this when implementing process() that returns results via futures
        instead of emitting to an output port.

        Args:
            max_pending: Max rows in flight (backpressure threshold)
            name: Name for logging/metrics (default: class name)
            max_workers: Worker threads (default: max_pending)
        """
        self._batch_name = name or self.__class__.__name__
        self._batch_output = None  # No output port for future-based API
        self._batch_shutdown = threading.Event()

        # Row reorder buffer
        self._batch_buffer = RowReorderBuffer(
            max_pending=max_pending,
            name=self._batch_name,
        )

        # Worker pool
        self._batch_executor = ThreadPoolExecutor(
            max_workers=max_workers or max_pending,
            thread_name_prefix=f"{self._batch_name}-worker",
        )

        # Future tracking
        self._batch_futures = {}
        self._batch_sequence_lock = threading.Lock()
        self._batch_next_sequence = 0

        # No release thread needed for future-based API
        # Results are consumed via future.result() instead
```

Then in azure.py:

```python
        # Batch processing initialization (future-based API)
        self.init_batch_processing_futures(
            max_pending=config.pool_size,
            name=f"azure-llm-{config.deployment_name}",
        )
        self._batch_initialized = True
```

**Step 5: Run test**

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure.py::test_azure_llm_process_api -xvs`

Expected: Test passes

**Step 6: Remove accept() and connect_output()**

Since we're unifying to `process()`, remove the `accept()` and `connect_output()` methods from `AzureLLMTransform`.

Delete lines 186-223 in `src/elspeth/plugins/llm/azure.py`.

**Step 7: Run full test suite for azure**

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure.py -xvs`

Expected: Some tests will fail because they use `accept()` - we'll fix those in Task 2.2

**Step 8: Commit**

```bash
git add src/elspeth/plugins/batching/mixin.py src/elspeth/plugins/llm/azure.py tests/plugins/llm/test_azure.py
git commit -m "feat(azure): implement process() using future-based API

- Add init_batch_processing_futures() to BatchTransformMixin
- Implement process() that blocks on future.result()
- Remove accept() and connect_output() methods
- Use configured timeout from LLMConfig

BREAKING: accept() API removed, use process() instead"
```

---

### Task 2.2: Update Azure LLM Tests to Use process()

**Files:**
- Modify: `tests/plugins/llm/test_azure.py`

**Step 1: Find all tests using accept()**

Run: `grep -n "\.accept(" tests/plugins/llm/test_azure.py`

Expected: List of line numbers where `.accept()` is called

**Step 2: Update first test using accept()**

For each test:
1. Replace `transform.accept(row, ctx)` with `result = transform.process(row, ctx)`
2. Remove `transform.flush_batch_processing()` calls
3. Verify result directly instead of collecting from output port

Example transformation:
```python
# Before:
output = CollectorOutputPort()
transform.connect_output(output, max_pending=10)
transform.accept(row, ctx)
transform.flush_batch_processing()
results = output.get_all_results()
assert len(results) == 1
token, result = results[0]

# After:
result = transform.process(row, ctx)
assert result.status == "success"
```

**Step 3: Run tests after each update**

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure.py -xvs`

Expected: Tests pass one by one as we update them

**Step 4: Commit**

```bash
git add tests/plugins/llm/test_azure.py
git commit -m "test(azure): update tests to use process() API

- Replace accept()/flush() pattern with direct process() calls
- Remove CollectorOutputPort usage (no longer needed)
- Verify results synchronously"
```

---

### Task 2.3: Update AzureMultiQueryLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_multi_query.py`
- Test: `tests/plugins/llm/test_azure_multi_query.py`

**Step 1: Read current implementation**

Read: `src/elspeth/plugins/llm/azure_multi_query.py` lines 390-400

Understand current `process()` implementation.

**Step 2: Replace process() implementation**

Similar to Task 2.1, replace the `NotImplementedError` with future-based implementation:

```python
    def process(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row with multiple LLM queries.

        Submits all queries for this row to the batch processing infrastructure
        and blocks until all queries complete. The 4 queries are processed
        concurrently within the worker thread.

        Args:
            row: Row to process
            ctx: Plugin context

        Returns:
            TransformResult with all query results added to row
        """
        future = self._submit_row(row, ctx, self._process_row)
        timeout = getattr(self._config, 'timeout', 120.0)
        return future.result(timeout=timeout)
```

**Step 3: Update __init__ to use init_batch_processing_futures()**

Similar changes to Task 2.1 Step 4.

**Step 4: Remove accept() and connect_output()**

**Step 5: Update tests**

Similar to Task 2.2, update all tests in `test_azure_multi_query.py` to use `process()`.

**Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure_multi_query.py -xvs`

Expected: All tests pass

**Step 7: Commit**

```bash
git add src/elspeth/plugins/llm/azure_multi_query.py tests/plugins/llm/test_azure_multi_query.py
git commit -m "feat(azure-multi): implement process() using future-based API"
```

---

### Task 2.4: Update OpenRouterLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter.py`
- Test: `tests/plugins/llm/test_openrouter.py`

**Step 1-7: Same pattern as Task 2.1 and 2.2**

Replace `process()`, update `__init__`, remove `accept()`, update tests.

**Step 8: Commit**

```bash
git add src/elspeth/plugins/llm/openrouter.py tests/plugins/llm/test_openrouter.py
git commit -m "feat(openrouter): implement process() using future-based API"
```

---

### Task 2.5: Update OpenRouterMultiQueryLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter_multi_query.py`
- Test: `tests/plugins/llm/test_openrouter_multi_query.py`

**Step 1-7: Same pattern as Task 2.3**

**Step 8: Commit**

```bash
git add src/elspeth/plugins/llm/openrouter_multi_query.py tests/plugins/llm/test_openrouter_multi_query.py
git commit -m "feat(openrouter-multi): implement process() using future-based API"
```

---

## Phase 3: Update Integration Tests

### Task 3.1: Update LLM Integration Tests

**Files:**
- Modify: `tests/integration/test_llm_transforms.py`
- Modify: `tests/integration/test_multi_query_integration.py`

**Step 1: Find integration tests using accept()**

Run: `grep -rn "\.accept(" tests/integration/`

**Step 2: Update each integration test**

Replace `accept()/flush()` patterns with `process()` calls.

**Step 3: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/test_llm_transforms.py tests/integration/test_multi_query_integration.py -xvs`

Expected: All pass

**Step 4: Commit**

```bash
git add tests/integration/
git commit -m "test(integration): update LLM integration tests to use process()"
```

---

## Phase 4: Verify Audit Trail Integration

### Task 4.1: Test TransformExecutor with Batch Transforms

**Files:**
- Test: `tests/engine/test_transform_executor_with_batch.py` (new file)

**Step 1: Write integration test**

Create `tests/engine/test_transform_executor_with_batch.py`:

```python
"""Test that TransformExecutor works correctly with batch transforms."""

def test_transform_executor_calls_batch_transform_process(test_landscape):
    """Verify TransformExecutor can call process() on batch transforms."""
    from elspeth.engine.executors import TransformExecutor
    from elspeth.engine.spans import SpanFactory
    from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig
    from elspeth.core.context import PluginContext
    from elspeth.core.token import TokenInfo

    # Create batch transform
    config = AzureOpenAIConfig(
        deployment_name="gpt-4",
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        pool_size=2,
        system_prompt="Classify",
        user_prompt_template="{{text}}",
        output_key="result",
    )
    transform = AzureLLMTransform(config=config)
    transform.node_id = "transform-1"

    # Create executor
    spans = SpanFactory(enabled=False)
    executor = TransformExecutor(recorder=test_landscape, span_factory=spans)

    # Create token
    token = TokenInfo(
        row_id=1,
        token_id=1,
        row_data={"text": "test"},
        branch_name=None,
    )
    ctx = PluginContext(token=token, landscape=test_landscape)

    # Execute through executor (this calls process())
    result, updated_token, error_sink = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=0,
    )

    # Verify result
    assert result.status == "success"
    assert error_sink is None

    # Verify audit trail was recorded
    states = test_landscape.get_node_states(token_id=token.token_id)
    assert len(states) == 1
    assert states[0].node_id == "transform-1"
    assert states[0].status == "completed"

    # Cleanup
    transform.close()
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/engine/test_transform_executor_with_batch.py -xvs`

Expected: Test passes (validates audit trail integration)

**Step 3: Commit**

```bash
git add tests/engine/test_transform_executor_with_batch.py
git commit -m "test(engine): verify TransformExecutor works with batch transforms

Validates that:
- executor.execute_transform() can call process() on batch transforms
- node_states are recorded correctly
- ctx.state_id is set for external call recording"
```

---

## Phase 5: Performance Validation

### Task 5.1: Benchmark Throughput

**Files:**
- Create: `tests/performance/test_batch_transform_throughput.py`

**Step 1: Write benchmark test**

```python
"""Benchmark batch transform throughput with process() API."""
import time

def test_process_api_throughput():
    """Verify process() API maintains concurrent throughput."""
    from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig
    from elspeth.core.context import PluginContext
    from elspeth.core.token import TokenInfo

    # Create transform with pool_size=10
    config = AzureOpenAIConfig(
        deployment_name="gpt-4",
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        pool_size=10,
        system_prompt="test",
        user_prompt_template="{{text}}",
        output_key="result",
    )
    transform = AzureLLMTransform(config=config)

    # Process 100 rows
    num_rows = 100
    start = time.perf_counter()

    for i in range(num_rows):
        token = TokenInfo(
            row_id=i,
            token_id=i,
            row_data={"text": f"row {i}"},
            branch_name=None,
        )
        ctx = PluginContext(token=token, landscape=None)
        result = transform.process({"text": f"row {i}"}, ctx)
        assert result.status == "success"

    duration = time.perf_counter() - start
    throughput = num_rows / duration

    print(f"\nProcessed {num_rows} rows in {duration:.2f}s ({throughput:.1f} rows/sec)")

    # With pool_size=10 and fast mocked LLM, expect > 50 rows/sec
    assert throughput > 50, f"Throughput too low: {throughput:.1f} rows/sec"

    transform.close()
```

**Step 2: Run benchmark**

Run: `.venv/bin/python -m pytest tests/performance/test_batch_transform_throughput.py -xvs`

Expected: Test passes, shows throughput metrics

**Step 3: Commit**

```bash
git add tests/performance/test_batch_transform_throughput.py
git commit -m "test(perf): add throughput benchmark for process() API"
```

---

## Phase 6: Cleanup and Documentation

### Task 6.1: Remove accept() from BatchTransformMixin

**Files:**
- Modify: `src/elspeth/plugins/batching/mixin.py`

**Step 1: Mark accept_row() as deprecated**

Since accept_row() might be used by other code, let's add a deprecation warning first:

```python
    def accept_row(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
        processor: Callable[[dict[str, Any], PluginContext], TransformResult],
    ) -> None:
        """DEPRECATED: Use _submit_row() for process() implementations.

        This method is for the old accept()/flush() API pattern.
        New code should implement process() using _submit_row().
        """
        import warnings
        warnings.warn(
            "accept_row() is deprecated. Implement process() using _submit_row() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # ... existing implementation ...
```

**Step 2: Update docstring**

Update class docstring to show process() pattern instead of accept() pattern.

**Step 3: Commit**

```bash
git add src/elspeth/plugins/batching/mixin.py
git commit -m "refactor(batching): deprecate accept_row() in favor of process()

The future-based _submit_row() API is now the recommended approach
for implementing process() in batch transforms."
```

---

### Task 6.2: Update Row-Level Pipelining Design Doc

**Files:**
- Modify: `docs/design/row-level-pipelining-design.md`

**Step 1: Read current design doc**

Read: `docs/design/row-level-pipelining-design.md`

**Step 2: Update to reflect process() API**

Update examples to show process() instead of accept():

```markdown
## Usage Pattern

### Before (Old API)
```python
transform.connect_output(output_port, max_pending=30)
for row in rows:
    transform.accept(row, ctx)
transform.flush_batch_processing()
```

### After (New API)
```python
for row in rows:
    result = transform.process(row, ctx)
    # result available immediately
```

## Architecture

The process() method:
1. Submits row to internal batch infrastructure via _submit_row()
2. Returns a Future[TransformResult]
3. Blocks on future.result() until row completes
4. Meanwhile, other process() calls can submit more work

This maintains concurrency while presenting a synchronous interface.
```

**Step 3: Commit**

```bash
git add docs/design/row-level-pipelining-design.md
git commit -m "docs: update row-level pipelining to show process() API"
```

---

### Task 6.3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add note about unified transform API**

Add section after "Transform Subtypes":

```markdown
### Transform API: process() is Universal

ALL transforms implement `process(row, ctx) -> TransformResult`, including
transforms that use concurrent batch processing internally.

**Batch transforms** (LLM transforms using `BatchTransformMixin`):
- Implement process() by submitting to internal thread pool
- Block on future.result() until that specific row completes
- Concurrency happens INSIDE the plugin, invisible to the engine
- From engine's perspective, process() is just "slow"

**Anti-pattern:**
```python
def process(self, row, ctx):
    raise NotImplementedError("Use accept() instead")
```

This violates the Liskov Substitution Principle. ALL TransformProtocol
implementations must have working process() methods.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document unified process() API in CLAUDE.md"
```

---

## Phase 7: Final Verification

### Task 7.1: Run Full Test Suite

**Step 1: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -x`

Expected: All tests pass

**Step 2: Check for remaining accept() usage**

Run: `grep -rn "\.accept(" src/elspeth/ tests/`

Expected: No matches in src/, only in deprecated tests or examples

**Step 3: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth`

Expected: No type errors

**Step 4: Final commit**

```bash
git add .
git commit -m "chore: final verification - all tests passing"
```

---

## Verification Checklist

- [ ] All transforms implement process() without NotImplementedError
- [ ] TransformExecutor works with batch transforms
- [ ] Audit trail records all batch transform operations (node_states)
- [ ] External calls recorded via ctx.state_id
- [ ] All existing tests pass with process() API
- [ ] No deadlocks under concurrent load (tested with 100+ rows)
- [ ] Performance acceptable (>50 rows/sec with pool_size=10)
- [ ] FIFO ordering preserved (verified via RowReorderBuffer)
- [ ] Exceptions propagate through futures correctly
- [ ] Timeout behavior works as expected
- [ ] accept() deprecated but not yet removed (for backwards compat)

---

## Rollback Plan

If critical issues discovered during implementation:

1. Revert to last known good commit
2. Keep both APIs temporarily (process() and accept())
3. Add feature flag to control which API is used
4. Investigate issue before completing migration

**Note:** Per CLAUDE.md "No Legacy Code Policy", we should NOT keep both APIs
long-term. The rollback plan is for emergencies during implementation only.
