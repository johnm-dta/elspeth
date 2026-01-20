# LLM Batch Aggregation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable LLM transforms (OpenRouter, Azure) to process rows in parallel batches using existing aggregation infrastructure with `output_mode: passthrough`.

**Architecture:** Set `is_batch_aware=True` on LLM transforms. When configured as aggregation nodes, engine buffers rows until trigger fires, then calls `process(rows: list[dict])`. PooledExecutor processes all rows in parallel. Results returned via `TransformResult.success_multi()`.

**Tech Stack:** Existing aggregation infrastructure, PooledExecutor, AIMD throttling, httpx.

---

## Task 1: Update RowContext Docstring

**Files:**
- Modify: `src/elspeth/plugins/llm/pooled_executor.py:28-43`

**Step 1: Update the docstring to clarify shared state_id is valid**

The current docstring says each row should have its "own" state_id. For batch aggregation, all rows share the same state_id (call_index provides uniqueness). Update to clarify this.

```python
@dataclass
class RowContext:
    """Context for processing a single row in the pool.

    Attributes:
        row: The row data to process
        state_id: State ID for audit trail. Can be unique per row OR shared
            across batch rows (when used with aggregation). When shared,
            call_index in PluginContext provides uniqueness for external_calls.
        row_index: Original index for result ordering
    """

    row: dict[str, Any]
    state_id: str
    row_index: int
```

**Step 2: Run tests to verify no regression**

Run: `pytest tests/plugins/llm/test_pooled_executor.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/elspeth/plugins/llm/pooled_executor.py
git commit -m "docs(llm): clarify RowContext state_id can be shared for batch"
```

---

## Task 2: Add Batch Support to OpenRouterLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter.py:43-65`
- Test: `tests/plugins/llm/test_openrouter.py`

**Step 1: Write the failing test for batch processing**

Add to `tests/plugins/llm/test_openrouter.py`:

```python
class TestOpenRouterBatchProcessing:
    """Tests for batch-aware aggregation processing."""

    @pytest.fixture
    def batch_config(self) -> dict[str, Any]:
        """Config with pooling enabled for batch processing."""
        return {
            "model": "anthropic/claude-3-haiku",
            "template": "Analyze: {{ row.text }}",
            "api_key": "test-key",
            "pool_size": 3,
            "schema": {"fields": "dynamic"},
        }

    @pytest.fixture
    def batch_transform(self, batch_config: dict[str, Any]) -> OpenRouterLLMTransform:
        """Create transform with batch config."""
        return OpenRouterLLMTransform(batch_config)

    def test_is_batch_aware_is_true(self, batch_transform: OpenRouterLLMTransform) -> None:
        """Transform should declare batch awareness for aggregation."""
        assert batch_transform.is_batch_aware is True

    def test_process_accepts_list_of_rows(
        self,
        batch_transform: OpenRouterLLMTransform,
        ctx: PluginContext,
        mock_http_client: Mock,
    ) -> None:
        """process() should accept list[dict] for batch aggregation."""
        # Mock successful responses for 3 rows
        mock_http_client.post.return_value = _create_mock_response(
            content="Sentiment: positive",
            status_code=200,
        )

        rows = [
            {"text": "I love this product!"},
            {"text": "This is terrible."},
            {"text": "It's okay I guess."},
        ]

        with patch.object(batch_transform, "_get_http_client", return_value=mock_http_client):
            result = batch_transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row is True
        assert len(result.rows) == 3
        # Each row should have response field
        for output_row in result.rows:
            assert "llm_response" in output_row

    def test_batch_with_partial_failures(
        self,
        batch_transform: OpenRouterLLMTransform,
        ctx: PluginContext,
        mock_http_client: Mock,
    ) -> None:
        """Batch should continue even if some rows fail (per-row error tracking)."""
        # First call succeeds, second fails, third succeeds
        mock_http_client.post.side_effect = [
            _create_mock_response(content="Result 1", status_code=200),
            _create_mock_response(
                status_code=400,
                raise_for_status_error=httpx.HTTPStatusError(
                    "Bad Request", request=Mock(), response=Mock(status_code=400)
                ),
            ),
            _create_mock_response(content="Result 3", status_code=200),
        ]

        rows = [
            {"text": "Row 1"},
            {"text": "Row 2 - will fail"},
            {"text": "Row 3"},
        ]

        with patch.object(batch_transform, "_get_http_client", return_value=mock_http_client):
            result = batch_transform.process(rows, ctx)

        # Should still succeed overall with per-row errors
        assert result.status == "success"
        assert result.is_multi_row is True
        assert len(result.rows) == 3

        # Row 0 and 2 should have responses
        assert result.rows[0]["llm_response"] is not None
        assert result.rows[2]["llm_response"] is not None

        # Row 1 should have error
        assert result.rows[1]["llm_response"] is None
        assert "llm_response_error" in result.rows[1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_openrouter.py::TestOpenRouterBatchProcessing::test_is_batch_aware_is_true -v`
Expected: FAIL with `AssertionError: assert False is True` (is_batch_aware not set)

**Step 3: Add is_batch_aware = True to the class**

Modify `src/elspeth/plugins/llm/openrouter.py` around line 61:

```python
class OpenRouterLLMTransform(BaseTransform):
    """LLM transform using OpenRouter API.
    ...
    """

    name = "openrouter_llm"
    is_batch_aware = True  # NEW: Enable aggregation buffering

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC
```

**Step 4: Run first test to verify it passes**

Run: `pytest tests/plugins/llm/test_openrouter.py::TestOpenRouterBatchProcessing::test_is_batch_aware_is_true -v`
Expected: PASS

**Step 5: Commit incremental progress**

```bash
git add src/elspeth/plugins/llm/openrouter.py tests/plugins/llm/test_openrouter.py
git commit -m "feat(llm): add is_batch_aware=True to OpenRouterLLMTransform"
```

---

## Task 3: Implement process() Polymorphism for OpenRouter

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter.py:121-150`

**Step 1: Modify process() to dispatch based on input type**

Update the `process()` method signature and add dispatch logic:

```python
def process(
    self,
    row: dict[str, Any] | list[dict[str, Any]],
    ctx: PluginContext,
) -> TransformResult:
    """Process row(s) via OpenRouter API.

    When is_batch_aware=True and used in aggregation, receives list[dict].
    Otherwise receives single dict.

    Args:
        row: Single row dict OR list of row dicts (batch aggregation)
        ctx: Plugin context with landscape and state_id

    Returns:
        TransformResult with processed row(s) or error
    """
    # Dispatch to batch processing if given a list
    if isinstance(row, list):
        return self._process_batch(row, ctx)

    # Route to pooled execution if configured (single row)
    if self._executor is not None:
        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError(
                "Pooled execution requires landscape recorder and state_id. "
                "Ensure transform is executed through the engine."
            )
        row_ctx = RowContext(row=row, state_id=ctx.state_id, row_index=0)
        try:
            results = self._executor.execute_batch(
                contexts=[row_ctx],
                process_fn=self._process_single_with_state,
            )
            return results[0]
        finally:
            with self._http_clients_lock:
                self._http_clients.pop(ctx.state_id, None)

    # Sequential execution path (existing code follows...)
    # ... rest of existing single-row implementation
```

**Step 2: Run tests to verify single-row path still works**

Run: `pytest tests/plugins/llm/test_openrouter.py::TestOpenRouterLLMTransformProcess -v`
Expected: All existing tests pass

**Step 3: Commit**

```bash
git add src/elspeth/plugins/llm/openrouter.py
git commit -m "refactor(llm): add polymorphic dispatch in OpenRouter.process()"
```

---

## Task 4: Implement _process_batch() for OpenRouter

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter.py` (add new method after process())

**Step 1: Implement the batch processing method**

Add after the `process()` method:

```python
def _process_batch(
    self,
    rows: list[dict[str, Any]],
    ctx: PluginContext,
) -> TransformResult:
    """Process batch of rows with parallel execution via PooledExecutor.

    Called when transform is used as aggregation node and trigger fires.
    All rows share the same state_id; call_index provides audit uniqueness.

    Args:
        rows: List of row dicts from aggregation buffer
        ctx: Plugin context with shared state_id for entire batch

    Returns:
        TransformResult.success_multi() with one output row per input
    """
    if not rows:
        return TransformResult.success({"batch_empty": True, "row_count": 0})

    if ctx.landscape is None or ctx.state_id is None:
        raise RuntimeError(
            "Batch processing requires landscape recorder and state_id. "
            "Ensure transform is executed through the engine."
        )

    # Ensure we have an executor for parallel processing
    if self._executor is None:
        # Fallback: process sequentially if no pool configured
        return self._process_batch_sequential(rows, ctx)

    # Create contexts - all rows share same state_id (call_index provides uniqueness)
    contexts = [
        RowContext(row=row, state_id=ctx.state_id, row_index=i)
        for i, row in enumerate(rows)
    ]

    # Execute all rows in parallel
    try:
        results = self._executor.execute_batch(
            contexts=contexts,
            process_fn=self._process_single_with_state,
        )
    finally:
        # Clean up cached clients
        with self._http_clients_lock:
            self._http_clients.pop(ctx.state_id, None)

    # Assemble output with per-row error tracking
    return self._assemble_batch_results(rows, results)

def _process_batch_sequential(
    self,
    rows: list[dict[str, Any]],
    ctx: PluginContext,
) -> TransformResult:
    """Fallback for batch processing without executor (pool_size=1)."""
    results: list[TransformResult] = []
    for row in rows:
        # Use existing sequential processing
        result = self._process_sequential(row, ctx)
        results.append(result)
    return self._assemble_batch_results(rows, results)

def _assemble_batch_results(
    self,
    rows: list[dict[str, Any]],
    results: list[TransformResult],
) -> TransformResult:
    """Assemble batch results with per-row error tracking.

    Follows AzureBatchLLMTransform pattern: include all rows in output,
    mark failures with {response_field}_error instead of failing entire batch.

    Args:
        rows: Original input rows
        results: TransformResults from processing each row

    Returns:
        TransformResult.success_multi() with one output per input
    """
    output_rows: list[dict[str, Any]] = []
    all_failed = True

    for i, (row, result) in enumerate(zip(rows, results)):
        output_row = dict(row)

        if result.success and result.row is not None:
            all_failed = False
            # Copy response fields from result
            output_row[self._response_field] = result.row.get(self._response_field)
            output_row[f"{self._response_field}_usage"] = result.row.get(
                f"{self._response_field}_usage"
            )
            output_row[f"{self._response_field}_template_hash"] = result.row.get(
                f"{self._response_field}_template_hash"
            )
            output_row[f"{self._response_field}_variables_hash"] = result.row.get(
                f"{self._response_field}_variables_hash"
            )
            output_row[f"{self._response_field}_template_source"] = result.row.get(
                f"{self._response_field}_template_source"
            )
            output_row[f"{self._response_field}_lookup_hash"] = result.row.get(
                f"{self._response_field}_lookup_hash"
            )
            output_row[f"{self._response_field}_lookup_source"] = result.row.get(
                f"{self._response_field}_lookup_source"
            )
            output_row[f"{self._response_field}_model"] = result.row.get(
                f"{self._response_field}_model"
            )
        else:
            # Per-row error tracking - don't fail entire batch
            output_row[self._response_field] = None
            output_row[f"{self._response_field}_error"] = result.error or {
                "reason": "unknown_error",
                "row_index": i,
            }

        output_rows.append(output_row)

    # Only return error if ALL rows failed
    if all_failed and output_rows:
        return TransformResult.error({
            "reason": "all_rows_failed",
            "row_count": len(rows),
        })

    return TransformResult.success_multi(output_rows)
```

**Step 2: Run batch tests to verify implementation**

Run: `pytest tests/plugins/llm/test_openrouter.py::TestOpenRouterBatchProcessing -v`
Expected: All batch tests pass

**Step 3: Run full test suite to verify no regressions**

Run: `pytest tests/plugins/llm/test_openrouter.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/elspeth/plugins/llm/openrouter.py
git commit -m "feat(llm): implement _process_batch() for OpenRouter aggregation"
```

---

## Task 5: Add Batch Support to AzureLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/azure.py`
- Test: `tests/plugins/llm/test_azure.py`

**Step 1: Apply same changes as OpenRouter**

The changes mirror Task 2-4:
1. Add `is_batch_aware = True` to class
2. Modify `process()` signature to accept `dict | list[dict]`
3. Add dispatch to `_process_batch()` at start of `process()`
4. Implement `_process_batch()`, `_process_batch_sequential()`, `_assemble_batch_results()`

**Step 2: Write tests mirroring OpenRouter batch tests**

Create `TestAzureBatchProcessing` class in `tests/plugins/llm/test_azure.py` with same test cases.

**Step 3: Run tests**

Run: `pytest tests/plugins/llm/test_azure.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/elspeth/plugins/llm/azure.py tests/plugins/llm/test_azure.py
git commit -m "feat(llm): add batch aggregation support to AzureLLMTransform"
```

---

## Task 6: Update OpenRouter Sentiment Example

**Files:**
- Create: `examples/openrouter_sentiment/settings_batched.yaml`
- Modify: `examples/openrouter_sentiment/README.md` (if exists)

**Step 1: Create batched settings file**

```yaml
# settings_batched.yaml
# Demonstrates batch aggregation with parallel LLM processing.
# Rows are buffered until trigger fires (10 rows), then processed in parallel.

source:
  plugin: csv_source
  options:
    path: data/reviews.csv
    schema:
      fields:
        - name: id
          type: string
        - name: text
          type: string

transforms:
  - plugin: openrouter_llm
    node_id: sentiment_batch  # Required for aggregation reference
    options:
      model: "anthropic/claude-3-haiku"
      api_key: "${OPENROUTER_API_KEY}"
      template: |
        Analyze the sentiment of this review. Respond with exactly one word:
        POSITIVE, NEGATIVE, or NEUTRAL.

        Review: {{ row.text }}
      response_field: sentiment
      pool_size: 5  # 5 parallel workers for batch processing
      schema:
        fields: dynamic

aggregations:
  - node: sentiment_batch
    trigger:
      type: COUNT
      threshold: 10  # Process in batches of 10 rows
    output_mode: passthrough  # N inputs → N outputs (each enriched)

sinks:
  - plugin: csv_sink
    options:
      path: output/sentiments_batched.csv
      schema:
        fields:
          - name: id
            type: string
          - name: text
            type: string
          - name: sentiment
            type: string
```

**Step 2: Commit**

```bash
git add examples/openrouter_sentiment/settings_batched.yaml
git commit -m "docs(examples): add batched settings for openrouter_sentiment"
```

---

## Task 7: Update Remaining LLM Examples

**Files:**
- Create: `examples/azure_llm/settings_batched.yaml` (if azure_llm example exists)
- Update any other LLM examples

**Step 1: Check which examples need updates**

Run: `ls -la examples/`

**Step 2: Add batched variants following same pattern as Task 6**

Each should demonstrate:
- `node_id` on transform
- `aggregations` section with trigger and `output_mode: passthrough`
- Comments explaining batch processing

**Step 3: Commit**

```bash
git add examples/
git commit -m "docs(examples): add batched settings to LLM examples"
```

---

## Task 8: Update No-Bug-Hiding Allowlist (if needed)

**Files:**
- Modify: `config/cicd/no_bug_hiding.yaml`

**Step 1: Run linting to check for new violations**

Run: `python -m ruff check src/elspeth/plugins/llm/openrouter.py`

**Step 2: If isinstance() checks trigger violations, add to allowlist**

The `isinstance(row, list)` check is a legitimate polymorphic dispatch, not bug-hiding. Add allowlist entry if needed:

```yaml
# Polymorphic dispatch for batch aggregation (dict vs list[dict])
- file: src/elspeth/plugins/llm/openrouter.py
  line: <line_number>
  pattern: isinstance
  reason: "Legitimate polymorphic dispatch for batch-aware process()"
```

**Step 3: Commit if changes needed**

```bash
git add config/cicd/no_bug_hiding.yaml
git commit -m "chore: update no-bug-hiding allowlist for batch dispatch"
```

---

## Task 9: Final Integration Test

**Files:**
- Test: Run full test suite

**Step 1: Run all LLM tests**

Run: `pytest tests/plugins/llm/ -v`
Expected: All tests pass

**Step 2: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/integration`
Expected: All tests pass

**Step 3: Run type checker**

Run: `python -m mypy src/elspeth/plugins/llm/`
Expected: No errors

**Step 4: Final commit if any fixes needed**

```bash
git add .
git commit -m "test: ensure all tests pass for LLM batch aggregation"
```

---

## Task 10: Push and Update PR

**Step 1: Push all commits**

```bash
git push origin multillm
```

**Step 2: Update PR description to mention batch aggregation feature**

The PR now includes:
- P2/P3 fixes from code review
- New batch aggregation support for LLM transforms
- Updated examples showing batched usage

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Update RowContext docstring | `pooled_executor.py` |
| 2 | Add `is_batch_aware=True` to OpenRouter | `openrouter.py` |
| 3 | Implement polymorphic `process()` dispatch | `openrouter.py` |
| 4 | Implement `_process_batch()` | `openrouter.py` |
| 5 | Apply same changes to Azure | `azure.py` |
| 6 | Update OpenRouter sentiment example | `examples/openrouter_sentiment/` |
| 7 | Update other LLM examples | `examples/` |
| 8 | Update allowlist if needed | `no_bug_hiding.yaml` |
| 9 | Final integration test | All test files |
| 10 | Push and update PR | Git |
