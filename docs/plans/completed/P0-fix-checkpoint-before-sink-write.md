# Implementation Plan: Fix Checkpoint Before Sink Write

**Bug:** P0-2026-01-19-checkpoint-before-sink-write.md
**Estimated Time:** 4-5 hours
**Complexity:** High
**Risk:** Medium-High (changes core checkpoint semantics)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.
>
> **Plan Review (2026-01-20):** Reviewed against codebase. Corrections applied:
> - Removed redundant Step 2 (sink_id_map already exists at line 474)
> - Added aggregation flush checkpoint handling to Step 1
> - Fixed line numbers for coalesce flush (was 816-822, actual 836-840)
> - Renumbered steps accordingly

## Summary

Checkpoints are created **before** sink writes, creating a durability gap:
1. Row processed through transforms → checkpoint created → token added to `pending_tokens`
2. (CRASH HERE = data loss)
3. Sink write happens later in batch

Recovery uses checkpoints to skip rows, but skipped rows may never have been written to their sink.

## Current Flow (Broken)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Main Processing Loop                             │
│                                                                      │
│  FOR each row:                                                       │
│    1. Process through transforms/gates                               │
│    2. Add token to pending_tokens[sink_name]                         │
│    3. _maybe_checkpoint(token_id, last_transform_node_id)  ← HERE!   │
│                                                                      │
│  AFTER all rows:                                                     │
│    4. FOR each sink:                                                 │
│         SinkExecutor.write(tokens)                         ← ACTUAL  │
└─────────────────────────────────────────────────────────────────────┘

CRASH WINDOW: Between step 3 and step 4
- Checkpoint says: "row was processed up to last_transform_node"
- Recovery says: "skip this row, it's checkpointed"
- Reality: row data never reached the sink
```

## Fixed Flow (Two Options)

### Option A: Checkpoint After Sink Write (Recommended)

Move checkpoint creation to AFTER successful sink write:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Main Processing Loop                             │
│                                                                      │
│  FOR each row:                                                       │
│    1. Process through transforms/gates                               │
│    2. Add token to pending_tokens[sink_name]                         │
│    (NO CHECKPOINT HERE)                                              │
│                                                                      │
│  AFTER all rows:                                                     │
│    3. FOR each sink:                                                 │
│         SinkExecutor.write(tokens)                                   │
│         FOR each token written:                                      │
│           _maybe_checkpoint(token_id, sink_node_id)        ← HERE!   │
└─────────────────────────────────────────────────────────────────────┘

CRASH WINDOW: None for already-written tokens
- Only checkpoint after sink confirms write
- Recovery correctly identifies unwritten rows
```

### Option B: Two-Phase Checkpoint (More Complex)

Add a "pending_sink" state that recovery understands:

```
┌─────────────────────────────────────────────────────────────────────┐
│  FOR each row:                                                       │
│    1. Process through transforms/gates                               │
│    2. Add token to pending_tokens[sink_name]                         │
│    3. checkpoint(token_id, status="pending_sink")          ← Phase 1 │
│                                                                      │
│  AFTER all rows:                                                     │
│    4. FOR each sink:                                                 │
│         SinkExecutor.write(tokens)                                   │
│         update_checkpoint(token_id, status="completed")    ← Phase 2 │
└─────────────────────────────────────────────────────────────────────┘

Recovery: Replay sink writes for "pending_sink" checkpoints
```

**Recommendation: Option A** - Simpler, no schema changes, clear semantics.

## Implementation Steps (Option A)

### Step 1: Remove in-loop checkpoints from orchestrator

**File:** `src/elspeth/engine/orchestrator.py`

**Location:** Lines 762-766, 773-777, 797-801 - remove `_maybe_checkpoint()` calls

```python
# REMOVE these blocks:

# Line 762-766 (after COMPLETED):
# self._maybe_checkpoint(
#     run_id=run_id,
#     token_id=result.token.token_id,
#     node_id=last_node_id,
# )

# Line 773-777 (after ROUTED):
# self._maybe_checkpoint(
#     run_id=run_id,
#     token_id=result.token.token_id,
#     node_id=last_node_id,
# )

# Line 797-801 (after COALESCED):
# self._maybe_checkpoint(
#     run_id=run_id,
#     token_id=result.token.token_id,
#     node_id=last_node_id,
# )
```

**ALSO:** Disable checkpointing in the aggregation flush helper call (line ~807):

```python
# Change this call to pass checkpoint=False:
agg_succeeded, agg_failed = self._flush_remaining_aggregation_buffers(
    config=config,
    processor=processor,
    aggregation_id_map=aggregation_id_map,
    ctx=ctx,
    pending_tokens=pending_tokens,
    output_sink_name=output_sink_name,
    run_id=run_id,
    checkpoint=False,  # ← Changed from True - checkpointing now happens after sink write
    last_node_id=default_last_node_id,
)
```

**Note:** The `_flush_remaining_aggregation_buffers` method has its own checkpointing logic at lines 1561-1566 and 1579-1584. By passing `checkpoint=False`, we defer checkpointing to the post-sink callback, ensuring aggregation batch tokens are also checkpointed only after they're durably written.

### Step 2: Add checkpoint callback to SinkExecutor.write()

**File:** `src/elspeth/engine/executors.py`

**Modify SinkExecutor.write() signature to accept optional checkpoint callback:**

```python
def write(
    self,
    sink: SinkProtocol,
    tokens: list[TokenInfo],
    ctx: PluginContext,
    step_in_pipeline: int,
    *,
    on_token_written: Callable[[TokenInfo], None] | None = None,
) -> Artifact | None:
    """Write tokens to sink with artifact recording.

    Args:
        sink: Sink plugin to write to
        tokens: Tokens to write (may be empty)
        ctx: Plugin context
        step_in_pipeline: Current position in DAG
        on_token_written: Optional callback called for each token after successful write.
                         Used for post-sink checkpointing.

    ...
    """
```

**After successful write, call the callback for each token:**

```python
# After sink.write() succeeds and node_states are completed:
if on_token_written is not None:
    for token in tokens:
        on_token_written(token)
```

### Step 3: Create checkpoint callback in orchestrator

**File:** `src/elspeth/engine/orchestrator.py`

**Location:** Before sink write loop (around line 850)

```python
# Create checkpoint callback for post-sink checkpointing
def checkpoint_after_sink(sink_node_id: str) -> Callable[[TokenInfo], None]:
    def callback(token: TokenInfo) -> None:
        self._maybe_checkpoint(
            run_id=run_id,
            token_id=token.token_id,
            node_id=sink_node_id,
        )
    return callback
```

### Step 4: Pass checkpoint callback to sink writes

**File:** `src/elspeth/engine/orchestrator.py`

**Location:** Sink write loop (lines 853-861)

**Note:** Use the existing `sink_id_map` (retrieved at line 474 via `graph.get_sink_id_map()`) - no need to create a new map.

```python
for sink_name, tokens in pending_tokens.items():
    if tokens and sink_name in config.sinks:
        sink = config.sinks[sink_name]
        sink_node_id = sink_id_map[sink_name]  # ← Use existing sink_id_map

        sink_executor.write(
            sink=sink,
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=step,
            on_token_written=checkpoint_after_sink(sink_node_id),
        )
```

### Step 5: Handle coalesce flush checkpoints

**File:** `src/elspeth/engine/orchestrator.py`

**Location:** Coalesce flush section (lines 821-846)

Remove the checkpoint calls from the coalesce flush loop. These will be checkpointed when the merged tokens are written to sink.

```python
# REMOVE lines 836-840:
# self._maybe_checkpoint(
#     run_id=run_id,
#     token_id=outcome.merged_token.token_id,
#     node_id=default_last_node_id,
# )
```

### Step 6: Update RunResult counter timing

**File:** `src/elspeth/engine/orchestrator.py`

The bug report mentions counters are incremented before sink write. This is actually OK - counters track processing outcomes, not sink durability. But we should document this distinction.

**Add comment clarifying counter semantics:**

```python
# Note: Counters track processing outcomes (how many rows reached each state).
# Sink durability is tracked separately via checkpoints, which are created
# AFTER successful sink writes. A crash before sink write means:
# - Counters may be inflated (row counted but not persisted)
# - But recovery will correctly identify the unwritten rows
```

### Step 7: Update checkpoint documentation

**File:** `src/elspeth/engine/orchestrator.py`

**Update `_maybe_checkpoint()` docstring:**

```python
def _maybe_checkpoint(self, run_id: str, token_id: str, node_id: str) -> None:
    """Create checkpoint if configured.

    Called after a token has been durably written to its terminal sink.
    The checkpoint represents a durable progress marker - recovery can
    safely skip any row whose token has a checkpoint with a sink node_id.

    IMPORTANT: Checkpoints are created AFTER sink writes, not during
    the main processing loop. This ensures the checkpoint represents
    actual durable output, not just processing completion.

    Args:
        run_id: Current run ID
        token_id: Token that was just written to sink
        node_id: Sink node that received the token
    """
```

### Step 8: Add integration test for crash recovery

**File:** `tests/engine/test_checkpoint_durability.py` (new file)

```python
"""Tests for checkpoint durability guarantees."""

import pytest

from elspeth.contracts import TokenInfo
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.checkpoint.recovery import RecoveryManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.orchestrator import Orchestrator


class TestCheckpointDurability:
    """Tests that checkpoints represent durable sink output."""

    def test_crash_before_sink_write_recovers_correctly(self, tmp_path):
        """Rows not yet written to sink should be reprocessed on recovery."""
        # Setup: Create a pipeline with a sink that crashes after N rows
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,value\n1,a\n2,b\n3,c\n4,d\n5,e\n")

        output_file = tmp_path / "output.csv"

        # Configure to crash after writing 2 rows
        crash_after = 2

        # First run: process all rows, crash during sink write
        # ... (setup orchestrator with crashing sink)

        # Verify: Only 2 rows were checkpointed (those written before crash)
        # ... (check checkpoint count)

        # Recovery: Resume should reprocess rows 3, 4, 5
        # ... (call resume, verify all 5 rows in output)

    def test_checkpoint_node_id_is_sink_not_transform(self, tmp_path):
        """Checkpoints should reference sink node, not last transform."""
        # Setup: Simple pipeline with one transform and one sink
        # Run: Process a row
        # Verify: Checkpoint node_id matches sink node, not transform node

    def test_no_checkpoint_for_pending_tokens(self, tmp_path):
        """Tokens added to pending_tokens but not yet written have no checkpoint."""
        # This tests the invariant that checkpoints only exist after sink write
```

### Step 9: Update existing tests

**File:** `tests/engine/test_orchestrator.py`

Some existing tests may assume checkpoints are created during the main loop. Update them to expect checkpoints after sink writes.

```python
# Tests that check checkpoint timing need updates:
# - Tests that verify checkpoint count during processing
# - Tests that mock _maybe_checkpoint and check call order
```

## Testing Checklist

- [ ] Checkpoints are created AFTER sink writes, not during main loop
- [ ] Checkpoint `node_id` is the sink node, not the last transform
- [ ] Crash before sink write → recovery reprocesses those rows
- [ ] Crash after sink write → recovery skips those rows (no duplicates)
- [ ] Coalesced tokens are checkpointed after sink write
- [ ] Routed tokens are checkpointed after their target sink write
- [ ] Forked tokens: each fork path checkpointed independently after its sink
- [ ] `on_token_written` callback is called for each token after sink.write()

## Run Tests

```bash
# Run new durability tests
.venv/bin/python -m pytest tests/engine/test_checkpoint_durability.py -v

# Run checkpoint-related tests
.venv/bin/python -m pytest tests/engine/ -k checkpoint -v

# Run recovery tests
.venv/bin/python -m pytest tests/engine/test_orchestrator_recovery.py -v

# Run full engine test suite
.venv/bin/python -m pytest tests/engine/ -v
```

## Acceptance Criteria

1. ✅ Checkpoints are created AFTER successful sink write
2. ✅ Checkpoint `node_id` references the sink node
3. ✅ Crash between processing and sink write → rows are reprocessed on recovery
4. ✅ No duplicate outputs for rows that were successfully checkpointed
5. ✅ Documentation updated to clarify checkpoint durability semantics

## Edge Cases

### Forked Rows
Each fork child reaches its own sink independently. Checkpoint after each child's sink write, using that sink's node_id.

### Aggregation Batches
Batch flush produces output token(s) that go to sink. Checkpoint the output token(s) after sink write, not during flush.

### Multiple Sinks
Different rows may route to different sinks. Each sink write triggers checkpoints for its tokens.

### Sink Write Failure
If `sink.write()` throws, no checkpoint is created (correct behavior - retry needed).

## Migration Notes

### Breaking Change: Checkpoint Timing

**Before:** Checkpoints created after transform processing, before sink write
**After:** Checkpoints created after sink write

**Impact:**
- Existing checkpoints from interrupted runs may be invalid (they may reference rows not yet written to sink)
- Recommendation: Clear checkpoints for any interrupted runs before upgrading

### Recovery Behavior Change

**Before:** Recovery might skip rows that were never written to sink
**After:** Recovery correctly identifies and reprocesses unwritten rows

This is a correctness improvement, not a breaking change in behavior.

## Security Notes

**Why this matters for audit integrity:**
- CLAUDE.md: "audit trail must withstand formal inquiry"
- A checkpoint should represent a verifiable, durable state
- Checkpointing before sink write creates "false completeness" - audit says done, but output doesn't exist
- This fix ensures checkpoints only exist for actually-persisted data

---

## Implementation Summary

**Status:** Completed
**Commits:** See git history for this feature
**Notes:** Checkpoint semantics corrected to create checkpoints after successful sink writes, ensuring crash recovery correctly identifies unwritten rows.
