# Implementation Plan: Aggregation Timeout Improvements

## Overview

Two remaining tasks from the 4-agent review of the aggregation timeout fix:

- **P2**: Refactor `_flush_remaining_aggregation_buffers` to use facades and honor output_mode
- **P3**: Add "true idle" capability (periodic timeout checks without row arrival)

---

## P2: Refactor `_flush_remaining_aggregation_buffers`

### Problem Statement

The `_flush_remaining_aggregation_buffers` method in `orchestrator.py` has two issues:

1. **Private member access**: Directly accesses `processor._aggregation_executor` instead of using public facade methods
2. **Output_mode bypass**: Same bug that was just fixed in `_check_aggregation_timeouts` - it bypasses `output_mode` semantics (passthrough/transform modes) and skips remaining transforms

### Current Code Issues

```python
# Line 2406: Private member access
buffered_count = processor._aggregation_executor.get_buffer_count(agg_node_id)

# Line 2432-2438: Private member access + wrong trigger type
flush_result, buffered_tokens, _batch_id = processor._aggregation_executor.execute_flush(
    node_id=agg_node_id,
    transform=agg_transform,
    ctx=ctx,
    step_in_pipeline=agg_step,
    trigger_type=TriggerType.END_OF_SOURCE,
)

# Lines 2441-2486: Direct routing to pending_tokens, bypasses output_mode
```

### Implementation Plan

#### Step 1: Extend `handle_timeout_flush` to support trigger_type parameter

**File**: `src/elspeth/engine/processor.py`

Modify `handle_timeout_flush` signature:

```python
def handle_timeout_flush(
    self,
    node_id: NodeID,
    transform: TransformProtocol,
    ctx: PluginContext,
    step: int,
    total_steps: int,
    trigger_type: TriggerType = TriggerType.TIMEOUT,  # NEW PARAMETER
) -> tuple[list[RowResult], list["_WorkItem"]]:
```

Update the internal `execute_flush` call to use the parameter:

```python
result, buffered_tokens, batch_id = self._aggregation_executor.execute_flush(
    node_id=node_id,
    transform=transform,
    ctx=ctx,
    step_in_pipeline=step,
    trigger_type=trigger_type,  # Use parameter instead of hardcoded TIMEOUT
)
```

#### Step 2: Update `_check_aggregation_timeouts` to pass trigger_type

**File**: `src/elspeth/engine/orchestrator.py`

Update the call to explicitly pass `TriggerType.TIMEOUT`:

```python
completed_results, work_items = processor.handle_timeout_flush(
    node_id=agg_node_id,
    transform=agg_transform,
    ctx=ctx,
    step=agg_step,
    total_steps=total_steps,
    trigger_type=TriggerType.TIMEOUT,  # Explicit
)
```

#### Step 3: Refactor `_flush_remaining_aggregation_buffers` to use facades

**File**: `src/elspeth/engine/orchestrator.py`

Replace the current implementation with:

```python
def _flush_remaining_aggregation_buffers(
    self,
    config: PipelineConfig,
    processor: RowProcessor,
    ctx: PluginContext,
    pending_tokens: dict[str, list[tuple[TokenInfo, RowOutcome | None]]],
    default_sink_name: str,
    run_id: str,
    recorder: LandscapeRecorder,
    checkpoint: bool = True,
    last_node_id: str | None = None,
) -> tuple[int, int]:
    """Flush remaining aggregation buffers at end-of-source."""
    rows_succeeded = 0
    rows_failed = 0
    total_steps = len(config.transforms)

    for agg_node_id_str, agg_settings in config.aggregation_settings.items():
        agg_node_id = NodeID(agg_node_id_str)

        # Use public facade (not private member)
        buffered_count = processor.get_aggregation_buffer_count(agg_node_id)
        if buffered_count == 0:
            continue

        # Find transform (existing logic)
        agg_transform, agg_step = self._find_aggregation_transform(
            config, agg_node_id_str, agg_settings.name
        )

        # Use handle_timeout_flush with END_OF_SOURCE trigger
        # This properly handles output_mode and routes through remaining transforms
        completed_results, work_items = processor.handle_timeout_flush(
            node_id=agg_node_id,
            transform=agg_transform,
            ctx=ctx,
            step=agg_step,
            total_steps=total_steps,
            trigger_type=TriggerType.END_OF_SOURCE,
        )

        # Handle completed results
        for result in completed_results:
            if result.outcome == RowOutcome.FAILED:
                rows_failed += 1
            else:
                sink_name = result.token.branch_name or default_sink_name
                if sink_name not in pending_tokens:
                    sink_name = default_sink_name
                pending_tokens[sink_name].append((result.token, result.outcome))
                rows_succeeded += 1

                if checkpoint and last_node_id is not None:
                    self._maybe_checkpoint(
                        run_id=run_id,
                        token_id=result.token.token_id,
                        node_id=last_node_id,
                    )

        # Process work items through remaining transforms
        for work_item in work_items:
            downstream_results = processor.process_token_from_step(
                token=work_item.token,
                transforms=config.transforms,
                ctx=ctx,
                start_step=work_item.start_step + 1,
            )

            for result in downstream_results:
                if result.outcome == RowOutcome.FAILED:
                    rows_failed += 1
                elif result.outcome == RowOutcome.COMPLETED:
                    sink_name = result.token.branch_name or default_sink_name
                    if sink_name not in pending_tokens:
                        sink_name = default_sink_name
                    pending_tokens[sink_name].append((result.token, result.outcome))
                    rows_succeeded += 1

                    if checkpoint and last_node_id is not None:
                        self._maybe_checkpoint(
                            run_id=run_id,
                            token_id=result.token.token_id,
                            node_id=last_node_id,
                        )

    return rows_succeeded, rows_failed
```

#### Step 4: Extract helper method for transform lookup (optional DRY improvement)

Create `_find_aggregation_transform` helper to avoid duplicate code between `_check_aggregation_timeouts` and `_flush_remaining_aggregation_buffers`.

### Testing

1. Run existing aggregation integration tests
2. Run full engine test suite
3. Verify passthrough and transform modes work at end-of-source

### Files Modified

- `src/elspeth/engine/processor.py` - Add `trigger_type` parameter to `handle_timeout_flush`
- `src/elspeth/engine/orchestrator.py` - Refactor `_flush_remaining_aggregation_buffers`

---

## P3: Add True Idle Timeout Capability

### Problem Statement

Currently, aggregation timeouts only fire when a new row arrives:

```
Row 1 buffered at T=0 (timeout=5s)
[No rows arrive for 60 seconds]
Row 2 arrives at T=60s → timeout check fires → flush happens

But the batch exceeded timeout at T=5, user expected flush then.
```

For streaming sources that may have long idle periods, the timeout should fire even without row arrivals.

### Design Options

#### Option A: Progress Loop Hook (Simple, 5s granularity)

Hook into the existing progress emission loop that runs every 5 seconds:

```python
if should_emit:
    # Existing progress emission...

    # NEW: Check aggregation timeouts during idle periods
    if config.aggregation_settings:
        timeout_succeeded, timeout_failed = self._check_aggregation_timeouts(...)
        rows_succeeded += timeout_succeeded
        rows_failed += timeout_failed
```

**Pros:**
- Minimal code changes
- No new threading or async complexity
- Uses existing infrastructure

**Cons:**
- 5-second granularity (timeouts may fire up to 5s late)
- Only works during active runs (progress loop runs while processing)

#### Option B: Source Iterator Timeout (Medium complexity)

Wrap the source iterator with a timeout-aware iterator that yields `None` when timeout expires:

```python
def _timeout_aware_source(self, source_iter, timeout_seconds):
    """Wrap source iterator with timeout-based yielding."""
    while True:
        try:
            # Wait for next row with timeout
            row = next_with_timeout(source_iter, timeout=timeout_seconds)
            yield row
        except TimeoutExpired:
            yield None  # Signal to check timeouts
        except StopIteration:
            break
```

**Pros:**
- Precise timeout granularity
- Works with any source

**Cons:**
- Requires threading or async for timeout on blocking iterators
- More complex implementation

#### Option C: Background Timer Thread (High complexity, precise)

Create a background thread that:
1. Sleeps until the next timeout deadline
2. Wakes up and signals the main loop to check timeouts
3. Recalculates next deadline after each check

```python
class TimeoutWatcher(threading.Thread):
    def __init__(self, processor, callback):
        self.processor = processor
        self.callback = callback
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            deadline = self.processor.get_next_timeout_deadline()
            if deadline is None:
                self._stop.wait(timeout=1.0)  # Check periodically
            else:
                sleep_time = deadline - time.monotonic()
                if sleep_time > 0:
                    self._stop.wait(timeout=sleep_time)
                if not self._stop.is_set():
                    self.callback()  # Signal main loop
```

**Pros:**
- Precise timeout firing
- No polling overhead

**Cons:**
- Thread synchronization complexity
- Harder to test
- Potential race conditions

### Recommended Approach: Option A (Progress Loop Hook)

For RC-1, Option A provides reasonable timeout behavior with minimal risk:

1. Most pipelines process rows continuously (never truly idle)
2. 5-second granularity is acceptable for most use cases
3. Critical latency requirements can use count triggers in addition to timeout
4. Can upgrade to Option B/C post-release if needed

### Implementation Plan for Option A

#### Step 1: Add timeout check to progress loop

**File**: `src/elspeth/engine/orchestrator.py`

In `_execute_run`, after the `should_emit` progress block:

```python
if should_emit:
    # ... existing progress emission code ...

    # Check aggregation timeouts during progress intervals
    # This provides 5-second granularity timeout checking during processing
    if config.aggregation_settings and agg_transform_lookup:
        timeout_succeeded, timeout_failed = self._check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=ctx,
            pending_tokens=pending_tokens,
            default_sink_name=default_sink_name,
            agg_transform_lookup=agg_transform_lookup,
        )
        rows_succeeded += timeout_succeeded
        rows_failed += timeout_failed
```

#### Step 2: Add facade for next timeout deadline (optional, for future Option B/C)

**File**: `src/elspeth/engine/processor.py`

```python
def get_next_timeout_deadline(self) -> float | None:
    """Get the earliest timeout deadline across all aggregations.

    Returns:
        Monotonic time of next timeout, or None if no timeouts pending
    """
    earliest: float | None = None
    for node_id, evaluator in self._aggregation_executor._trigger_evaluators.items():
        if evaluator.batch_count > 0 and evaluator._config.timeout_seconds is not None:
            deadline = evaluator._batch_start_time + evaluator._config.timeout_seconds
            if earliest is None or deadline < earliest:
                earliest = deadline
    return earliest
```

#### Step 3: Document the behavior

Update `CLAUDE.md` to note that true idle timeout has 5-second granularity in RC-1.

### Testing

1. Create test with slow source (yields row, sleeps 10s, yields row)
2. Configure aggregation with timeout=2s
3. Verify flush happens during the 10s sleep (within 5s window)

### Future Improvements (Post RC-1)

If precise idle timeouts become a requirement:

1. Implement Option B (source iterator timeout) for synchronous sources
2. Implement Option C (background thread) for maximum precision
3. Add `TriggerEvaluator.get_deadline()` method for deadline calculation

---

## Implementation Order

1. **P2 first** - This is a correctness fix (output_mode bypass) masquerading as a refactor
2. **P3 second** - This is a new capability

## Risk Assessment

| Task | Risk | Mitigation |
|------|------|------------|
| P2 | Low | Same pattern as `_check_aggregation_timeouts` fix already tested |
| P3 Option A | Low | Hook into existing loop, minimal changes |
| P3 Option B/C | Medium | Thread safety, complex testing - defer to post-RC1 |

## Estimated Effort

- P2: ~30 minutes (pattern already established)
- P3 Option A: ~20 minutes (simple hook)
- P3 Option B/C: ~4 hours (threading, testing)

---

## Implementation Status (2026-01-28)

### P2: COMPLETE ✅

**Changes Made:**

1. **`src/elspeth/engine/processor.py`**
   - Added `trigger_type: TriggerType` as **required** parameter to `handle_timeout_flush()`
   - Updated docstring to reflect support for both TIMEOUT and END_OF_SOURCE triggers

2. **`src/elspeth/engine/orchestrator.py`**
   - Added `_find_aggregation_transform()` helper method (DRY refactor)
   - Updated `_check_aggregation_timeouts()` to pass `trigger_type=TriggerType.TIMEOUT` explicitly
   - **Refactored `_flush_remaining_aggregation_buffers()`:**
     - Uses `processor.get_aggregation_buffer_count()` (public facade) instead of `processor._aggregation_executor.get_buffer_count()`
     - Uses `processor.handle_timeout_flush()` with `trigger_type=TriggerType.END_OF_SOURCE`
     - Correctly handles all output_modes (single, passthrough, transform)
     - Routes tokens through remaining transforms via `process_token_from_step()`

3. **`tests/engine/test_aggregation_integration.py`**
   - Added `TestEndOfSourceFlush` test class with 5 integration tests:
     - `test_end_of_source_single_mode` - verifies single output_mode
     - `test_end_of_source_passthrough_mode` - verifies passthrough preserves tokens
     - `test_end_of_source_transform_mode` - verifies N→M transformation
     - `test_end_of_source_passthrough_with_downstream_transform` - verifies downstream routing
     - `test_end_of_source_single_with_downstream_transform` - verifies downstream routing

**Test Results:** All 612 engine tests pass.

### P3: DEFERRED (Post RC-1) ⏸️

**Reason:** The 4-agent review identified a critical flaw in Option A (progress loop hook):
- The progress loop only runs while the source iterator is being consumed
- For blocking sources (e.g., message queues that wait for messages), the loop never runs
- This would create false confidence that timeouts work during "true idle" periods

**Future Work:** For streaming sources requiring precise idle timeouts:
- Option B (source iterator timeout) requires threading or async for blocking iterators
- Option C (background timer thread) provides maximum precision but adds complexity
- Both options deferred to post-RC1 based on actual user requirements
