# LLM Batch Aggregation Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable LLM transforms (OpenRouter, Azure) to process rows in parallel batches using existing aggregation infrastructure.

**Architecture:** LLM transforms become batch-aware (`is_batch_aware=True`). When configured as aggregation nodes with triggers, the engine buffers rows until trigger fires, then passes the batch to `process()`. The `PooledExecutor` processes all rows in parallel, and results are returned as enriched rows.

**Tech Stack:** Existing aggregation infrastructure, PooledExecutor, AIMD throttling.

---

## Problem Statement

The P2 review finding identified that `pool_size` configuration doesn't actually parallelize rows because:
1. The engine calls `process(row)` one row at a time
2. `execute_batch([single_row])` always processes single-row batches
3. The pool never has multiple rows to work on simultaneously

## Solution: Aggregation-Based Batching

Use existing aggregation infrastructure to buffer rows, then process the batch in parallel:

1. **Buffering**: Engine buffers rows until trigger fires (COUNT, SIZE, TIME)
2. **Parallel execution**: `PooledExecutor.execute_batch()` processes all rows concurrently
3. **Per-row results**: Each input row gets its own output row with LLM response or error

## Configuration

```yaml
transforms:
  - plugin: openrouter_llm
    node_id: sentiment_batch
    options:
      model: "anthropic/claude-3-haiku"
      template: "Analyze sentiment: {{ row.text }}"
      pool_size: 5  # Parallel workers for batch processing
      schema:
        fields: dynamic

aggregations:
  - node: sentiment_batch
    trigger:
      type: COUNT
      threshold: 10  # Process in batches of 10 rows
    output_mode: passthrough  # N inputs → N outputs (each enriched)
```

**Key configuration points:**
- `node_id` enables the transform to be referenced by aggregation config
- `pool_size` means "parallel workers within a batch"
- `output_mode: passthrough` preserves 1:1 input/output correspondence
- Existing trigger types work: COUNT, SIZE, TIME

## Transform Implementation

### Changes to OpenRouterLLMTransform and AzureLLMTransform

```python
class OpenRouterLLMTransform(BaseTransform):
    name = "openrouter_llm"
    is_batch_aware = True  # Signal engine to use aggregation buffering

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        if isinstance(row, list):
            return self._process_batch(row, ctx)
        return self._process_single_row(row, ctx)

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with parallel execution via PooledExecutor."""
        if not rows:
            return TransformResult.success({"batch_empty": True, "row_count": 0})

        # All rows use SAME state_id - call_index distinguishes individual calls
        contexts = [
            RowContext(row=row, state_id=ctx.state_id, row_index=i)
            for i, row in enumerate(rows)
        ]

        results = self._executor.execute_batch(
            contexts=contexts,
            process_fn=self._process_single_with_state,
        )

        # Per-row error tracking (following AzureBatchLLMTransform pattern)
        output_rows: list[dict[str, Any]] = []
        for i, (row, result) in enumerate(zip(rows, results)):
            output_row = dict(row)
            if result.success:
                output_row[self._response_field] = result.row[self._response_field]
                output_row[f"{self._response_field}_usage"] = result.row.get(
                    f"{self._response_field}_usage"
                )
                output_row[f"{self._response_field}_template_hash"] = result.row.get(
                    f"{self._response_field}_template_hash"
                )
                output_row[f"{self._response_field}_model"] = result.row.get(
                    f"{self._response_field}_model"
                )
            else:
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = result.error
            output_rows.append(output_row)

        return TransformResult.success_multi(output_rows)
```

## Error Handling

| Error Type | Behavior | Example |
|------------|----------|---------|
| Template error | Mark row with error, continue batch | Missing variable in Jinja2 |
| Capacity error (429/503/529) | PooledExecutor retries with AIMD backoff | Rate limit hit |
| API error (400/401/500) | Mark row with error, continue batch | Invalid request |
| Network error | Mark row with error, continue batch | Connection timeout |

**Output structure for failed rows:**
```python
{
    "original_field": "original value",
    "llm_response": None,
    "llm_response_error": {
        "reason": "api_call_failed",
        "error": "HTTP 400: Invalid request",
    }
}
```

**Batch-level failure:** Only if ALL rows fail, return `TransformResult.error()`.

## Audit Trail

- Single `state_id` for entire batch flush (created by `AggregationExecutor.execute_flush()`)
- Individual LLM calls distinguished by `call_index` (auto-incremented by `PluginContext.record_call()`)
- Unique constraint: `(state_id, call_index)` in `external_calls` table

## Limitations

1. **No checkpoint support**: PooledExecutor provides no crash recovery. For resilient batch processing with crash recovery, use `AzureBatchLLMTransform` which uses Azure's async Batch API.

2. **Synchronous execution**: Batch completes when all rows complete. Long-running batches block the pipeline.

## Example Updates Required

Update 3 LLM examples to show batched usage:
1. `examples/openrouter_sentiment/` - add `settings_batched.yaml`
2. `examples/azure_llm/` - add batched variant
3. Third LLM example - add batched variant

## Design Decisions

1. **Why aggregation infrastructure?** Reuses proven buffering, triggers, and flush logic.

2. **Why output_mode: passthrough?** Preserves 1:1 input/output correspondence (N→N).

3. **Why per-row error tracking?** Follows Three-Tier Trust Model - row data can fail without being a bug.

4. **Why shared state_id?** Engine creates one node_state per flush. call_index provides uniqueness.
