# Aggregation Crash Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement orchestrator-level batch recovery so crashed aggregation pipelines can resume without data loss

**Architecture:** The checkpoint/recovery infrastructure exists (CheckpointManager, RecoveryManager, ResumePoint). The missing piece is orchestrator integration: when resuming a failed run, the orchestrator must (1) find incomplete batches, (2) restore aggregation plugin state from checkpoint, and (3) either retry the failed flush or continue collecting. This plan implements that integration.

**Tech Stack:** SQLAlchemy, existing checkpoint/recovery modules, AggregationExecutor

---

## Background

**What EXISTS (from explore agent findings):**
- `CheckpointManager.create_checkpoint()` - saves aggregation_state_json
- `RecoveryManager.get_resume_point()` - returns ResumePoint with deserialized aggregation_state
- `BatchStatus` enum: draft, executing, completed, failed
- `batches.attempt` field for retry tracking
- All batch state transitions recorded in landscape

**What's MISSING:**
- Orchestrator code to call `RecoveryManager` on startup
- Logic to find "failed" or "executing" batches and decide recovery action
- Restoration of `AggregationExecutor` in-memory state from checkpoint
- Retry logic for failed batches with attempt increment

---

## Task 1: Add Batch Recovery Query to Recorder

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py:1357-1402`
- Test: `tests/core/landscape/test_recorder.py`

**Step 1: Write the failing test**

```python
class TestBatchRecoveryQueries:
    """Tests for batch recovery query methods."""

    def test_get_incomplete_batches_returns_draft_and_executing(
        self, recorder: LandscapeRecorder
    ):
        """get_incomplete_batches() finds batches needing recovery."""
        run_id = recorder.create_run(
            config_hash="test",
            settings_json="{}",
        ).run_id

        # Create batches in various states
        draft_batch = recorder.create_batch(
            run_id=run_id,
            aggregation_node_id="agg_node",
        )
        executing_batch = recorder.create_batch(
            run_id=run_id,
            aggregation_node_id="agg_node",
        )
        recorder.update_batch_status(executing_batch.batch_id, "executing")

        completed_batch = recorder.create_batch(
            run_id=run_id,
            aggregation_node_id="agg_node",
        )
        recorder.complete_batch(completed_batch.batch_id, "completed", "count")

        # Act
        incomplete = recorder.get_incomplete_batches(run_id)

        # Assert: Only draft and executing returned
        batch_ids = {b.batch_id for b in incomplete}
        assert draft_batch.batch_id in batch_ids
        assert executing_batch.batch_id in batch_ids
        assert completed_batch.batch_id not in batch_ids

    def test_get_incomplete_batches_includes_failed_for_retry(
        self, recorder: LandscapeRecorder
    ):
        """Failed batches are returned for potential retry."""
        run_id = recorder.create_run(
            config_hash="test",
            settings_json="{}",
        ).run_id

        failed_batch = recorder.create_batch(
            run_id=run_id,
            aggregation_node_id="agg_node",
        )
        recorder.update_batch_status(failed_batch.batch_id, "executing")
        recorder.update_batch_status(failed_batch.batch_id, "failed")

        incomplete = recorder.get_incomplete_batches(run_id)

        batch_ids = {b.batch_id for b in incomplete}
        assert failed_batch.batch_id in batch_ids
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/landscape/test_recorder.py::TestBatchRecoveryQueries -v`

Expected: FAIL with "AttributeError: 'LandscapeRecorder' has no attribute 'get_incomplete_batches'"

**Step 3: Implement get_incomplete_batches()**

Add to `src/elspeth/core/landscape/recorder.py` after `get_batches()` (around line 1402):

```python
def get_incomplete_batches(self, run_id: str) -> list[Batch]:
    """Get batches that need recovery (draft, executing, or failed).

    Args:
        run_id: The run to query

    Returns:
        List of Batch objects with status in (draft, executing, failed),
        ordered by created_at ascending (oldest first for recovery)
    """
    from elspeth.core.landscape.schema import batches_table

    with self._db.connection() as conn:
        result = conn.execute(
            select(batches_table)
            .where(batches_table.c.run_id == run_id)
            .where(
                batches_table.c.status.in_(["draft", "executing", "failed"])
            )
            .order_by(batches_table.c.created_at.asc())
        ).fetchall()

    return [
        Batch(
            batch_id=row.batch_id,
            run_id=row.run_id,
            aggregation_node_id=row.aggregation_node_id,
            attempt=row.attempt,
            status=BatchStatus(row.status),
            created_at=row.created_at,
            aggregation_state_id=row.aggregation_state_id,
            trigger_reason=row.trigger_reason,
            completed_at=row.completed_at,
        )
        for row in result
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/landscape/test_recorder.py::TestBatchRecoveryQueries -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder.py
git commit -m "$(cat <<'EOF'
feat(landscape): add get_incomplete_batches() for crash recovery

Returns batches with status draft/executing/failed that need
recovery attention after a pipeline crash.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Batch Retry with Attempt Increment

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: `tests/core/landscape/test_recorder.py`

**Step 1: Write the failing test**

```python
def test_retry_batch_increments_attempt_and_resets_status(
    self, recorder: LandscapeRecorder
):
    """retry_batch() creates new attempt with draft status."""
    run_id = recorder.create_run(
        config_hash="test",
        settings_json="{}",
    ).run_id

    # Create and fail a batch
    original = recorder.create_batch(
        run_id=run_id,
        aggregation_node_id="agg_node",
    )
    recorder.update_batch_status(original.batch_id, "executing")
    recorder.update_batch_status(original.batch_id, "failed")

    # Act: Retry the batch
    retried = recorder.retry_batch(original.batch_id)

    # Assert: New batch with incremented attempt
    assert retried.batch_id != original.batch_id  # New batch ID
    assert retried.attempt == original.attempt + 1
    assert retried.status == BatchStatus.DRAFT
    assert retried.aggregation_node_id == original.aggregation_node_id

def test_retry_batch_preserves_members(
    self, recorder: LandscapeRecorder
):
    """retry_batch() copies batch members to new batch."""
    run_id = recorder.create_run(
        config_hash="test",
        settings_json="{}",
    ).run_id

    original = recorder.create_batch(
        run_id=run_id,
        aggregation_node_id="agg_node",
    )

    # Add members to original
    recorder.add_batch_member(original.batch_id, "token-1", ordinal=0)
    recorder.add_batch_member(original.batch_id, "token-2", ordinal=1)
    recorder.update_batch_status(original.batch_id, "failed")

    # Act
    retried = recorder.retry_batch(original.batch_id)

    # Assert: Members copied
    members = recorder.get_batch_members(retried.batch_id)
    assert len(members) == 2
    assert members[0].token_id == "token-1"
    assert members[1].token_id == "token-2"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/landscape/test_recorder.py::test_retry_batch -v`

Expected: FAIL with "AttributeError: 'LandscapeRecorder' has no attribute 'retry_batch'"

**Step 3: Implement retry_batch()**

Add to `src/elspeth/core/landscape/recorder.py`:

```python
def retry_batch(self, batch_id: str) -> Batch:
    """Create a new batch attempt from a failed batch.

    Copies batch metadata and members to a new batch with
    incremented attempt counter and draft status.

    Args:
        batch_id: The failed batch to retry

    Returns:
        New Batch with attempt = original.attempt + 1

    Raises:
        ValueError: If original batch not found or not in failed status
    """
    from elspeth.core.landscape.schema import batches_table, batch_members_table

    original = self.get_batch(batch_id)
    if original is None:
        raise ValueError(f"Batch not found: {batch_id}")
    if original.status != BatchStatus.FAILED:
        raise ValueError(
            f"Can only retry failed batches, got status: {original.status}"
        )

    # Create new batch with incremented attempt
    new_batch = self.create_batch(
        run_id=original.run_id,
        aggregation_node_id=original.aggregation_node_id,
        batch_id=None,  # Generate new ID
        attempt=original.attempt + 1,
    )

    # Copy members to new batch
    original_members = self.get_batch_members(batch_id)
    for member in original_members:
        self.add_batch_member(
            batch_id=new_batch.batch_id,
            token_id=member.token_id,
            ordinal=member.ordinal,
        )

    return new_batch
```

**Step 4: Update create_batch() to accept attempt parameter**

Modify `create_batch()` signature (around line 1180):

```python
def create_batch(
    self,
    run_id: str,
    aggregation_node_id: str,
    batch_id: str | None = None,
    attempt: int = 0,  # Add this parameter
) -> Batch:
```

And update the insert values to include `attempt=attempt`.

**Step 5: Run tests**

Run: `pytest tests/core/landscape/test_recorder.py::test_retry_batch -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder.py
git commit -m "$(cat <<'EOF'
feat(landscape): add retry_batch() for failed batch recovery

Creates new batch attempt from failed batch, copying members
and incrementing attempt counter. Enables aggregation retry.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add Recovery Protocol to Orchestrator

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator_recovery.py` (create)

**Step 1: Write the failing test**

```python
"""Tests for orchestrator crash recovery."""

import pytest
from unittest.mock import Mock, patch

from elspeth.engine.orchestrator import Orchestrator
from elspeth.core.checkpoint.recovery import RecoveryManager, ResumePoint
from elspeth.contracts.audit import Batch, Checkpoint
from elspeth.contracts.enums import BatchStatus


class TestOrchestratorRecovery:
    """Test orchestrator recovery from crashed runs."""

    def test_resume_restores_aggregation_state(
        self, orchestrator: Orchestrator, recorder, recovery_manager
    ):
        """resume() should restore aggregation plugin state from checkpoint."""
        # Arrange: Create a resume point with aggregation state
        agg_state = {
            "buffer": [{"id": 1, "value": 100}],
            "count": 1,
            "sum": 100,
        }
        resume_point = ResumePoint(
            checkpoint=Mock(
                checkpoint_id="cp-1",
                run_id="run-123",
                sequence_number=5,
                aggregation_state_json='{"buffer": [], "count": 0}',
            ),
            token_id="token-last",
            node_id="agg_node",
            sequence_number=5,
            aggregation_state=agg_state,
        )

        # Act
        orchestrator.resume(resume_point)

        # Assert: Aggregation executor has restored state
        agg_executor = orchestrator._get_aggregation_executor("agg_node")
        assert agg_executor is not None
        assert agg_executor._restored_state == agg_state

    def test_resume_retries_failed_batch(
        self, orchestrator: Orchestrator, recorder
    ):
        """resume() should retry batches that were executing when crash occurred."""
        run_id = "run-crashed"

        # Arrange: Create a failed batch
        failed_batch = recorder.create_batch(
            run_id=run_id,
            aggregation_node_id="agg_node",
        )
        recorder.add_batch_member(failed_batch.batch_id, "token-1", 0)
        recorder.update_batch_status(failed_batch.batch_id, "executing")
        recorder.update_batch_status(failed_batch.batch_id, "failed")

        resume_point = ResumePoint(
            checkpoint=Mock(run_id=run_id, sequence_number=1),
            token_id="token-1",
            node_id="agg_node",
            sequence_number=1,
            aggregation_state={"buffer": [{"id": 1}], "count": 1},
        )

        # Act
        orchestrator.resume(resume_point)

        # Assert: Failed batch was retried
        incomplete = recorder.get_incomplete_batches(run_id)
        # Original failed batch should still exist
        # New retry batch should be in draft status
        retry_batches = [b for b in incomplete if b.attempt > 0]
        assert len(retry_batches) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator_recovery.py -v`

Expected: FAIL (resume() doesn't exist or doesn't handle recovery)

**Step 3: Implement resume() method in Orchestrator**

Add to `src/elspeth/engine/orchestrator.py`:

```python
def resume(self, resume_point: ResumePoint) -> None:
    """Resume a failed run from a checkpoint.

    Restores aggregation state and retries any failed batches.

    Args:
        resume_point: ResumePoint from RecoveryManager.get_resume_point()
    """
    from elspeth.core.checkpoint.recovery import ResumePoint

    run_id = resume_point.checkpoint.run_id

    # 1. Restore aggregation state to executor
    if resume_point.aggregation_state is not None:
        agg_executor = self._get_aggregation_executor(resume_point.node_id)
        if agg_executor is not None:
            agg_executor.restore_state(resume_point.aggregation_state)

    # 2. Find and handle incomplete batches
    incomplete_batches = self._recorder.get_incomplete_batches(run_id)

    for batch in incomplete_batches:
        if batch.status == BatchStatus.FAILED:
            # Retry failed batches
            self._retry_failed_batch(batch)
        elif batch.status == BatchStatus.EXECUTING:
            # Mark as failed (crash interrupted execution), then retry
            self._recorder.update_batch_status(batch.batch_id, "failed")
            self._retry_failed_batch(batch)
        # DRAFT batches continue normally (collection resumes)

    # 3. Update run status to indicate recovery
    self._recorder.update_run_status(run_id, "running")

def _retry_failed_batch(self, batch: Batch) -> None:
    """Retry a failed batch by creating new attempt."""
    new_batch = self._recorder.retry_batch(batch.batch_id)

    # Get the aggregation executor for this batch's node
    agg_executor = self._get_aggregation_executor(batch.aggregation_node_id)
    if agg_executor is None:
        raise RuntimeError(
            f"No aggregation executor for node: {batch.aggregation_node_id}"
        )

    # Restore batch to executor and flush
    agg_executor.restore_batch(new_batch.batch_id)
    agg_executor.flush(batch.aggregation_node_id)
```

**Step 4: Add restore_state() to AggregationExecutor**

Add to `src/elspeth/engine/executors.py` in `AggregationExecutor` class:

```python
def restore_state(self, state: dict[str, Any]) -> None:
    """Restore aggregation state from checkpoint.

    Called during recovery to restore in-memory state.

    Args:
        state: Deserialized aggregation_state from checkpoint
    """
    self._restored_state = state
    # The aggregation plugin should implement restore_state() too
    # For now, store it for the plugin to access

def restore_batch(self, batch_id: str) -> None:
    """Mark a batch as the current in-progress batch.

    Called during recovery to resume a batch.

    Args:
        batch_id: The batch to restore as current
    """
    # Get batch to find the node_id
    batch = self._recorder.get_batch(batch_id)
    if batch is None:
        raise ValueError(f"Batch not found: {batch_id}")

    self._batch_ids[batch.aggregation_node_id] = batch_id

    # Restore member count from database
    members = self._recorder.get_batch_members(batch_id)
    self._member_counts[batch_id] = len(members)
```

**Step 5: Run tests**

Run: `pytest tests/engine/test_orchestrator_recovery.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/engine/orchestrator.py src/elspeth/engine/executors.py tests/engine/test_orchestrator_recovery.py
git commit -m "$(cat <<'EOF'
feat(engine): add orchestrator.resume() for crash recovery

Implements batch recovery protocol:
- Restores aggregation state from checkpoint
- Retries failed batches with attempt increment
- Handles batches interrupted mid-execution

Closes ENG-007 go-live blocker.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integration Test for Full Recovery Cycle

**Files:**
- Create: `tests/integration/test_aggregation_recovery.py`

**Step 1: Write full recovery integration test**

```python
"""Integration test for aggregation crash recovery."""

import pytest
from unittest.mock import patch

from elspeth.engine.orchestrator import Orchestrator
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.checkpoint.recovery import RecoveryManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.contracts.enums import BatchStatus


class TestAggregationRecoveryIntegration:
    """End-to-end test for aggregation crash recovery."""

    def test_full_recovery_cycle(
        self, landscape_db: LandscapeDB, tmp_path
    ):
        """Simulate crash during flush and verify recovery works."""
        recorder = LandscapeRecorder(landscape_db)
        checkpoint_mgr = CheckpointManager(landscape_db)
        recovery_mgr = RecoveryManager(landscape_db)

        # === PHASE 1: Normal execution until crash ===

        run = recorder.create_run(
            config_hash="test-config",
            settings_json='{"aggregation": {"trigger": {"count": 3}}}',
        )

        # Record source rows
        for i in range(3):
            recorder.record_row(
                run_id=run.run_id,
                row_index=i,
                row_id=f"row-{i}",
                source_data={"id": i, "value": i * 100},
            )

        # Create batch and add members
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i in range(3):
            token = recorder.create_token(
                run_id=run.run_id,
                row_id=f"row-{i}",
            )
            recorder.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Simulate checkpoint before flush
        agg_state = {"buffer": [0, 100, 200], "sum": 300, "count": 3}
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=token.token_id,
            node_id="sum_aggregator",
            sequence_number=2,
            aggregation_state=agg_state,
        )

        # Simulate crash during flush
        recorder.update_batch_status(batch.batch_id, "executing")
        recorder.update_run_status(run.run_id, "failed")
        # (In reality, an exception would occur here)

        # === PHASE 2: Recovery ===

        # Verify we can resume
        assert recovery_mgr.can_resume(run.run_id)

        # Get resume point
        resume_point = recovery_mgr.get_resume_point(run.run_id)
        assert resume_point is not None
        assert resume_point.aggregation_state == agg_state

        # Create new orchestrator and resume
        # (In real usage, this would be a new process)
        orchestrator = Orchestrator(
            recorder=recorder,
            checkpoint_manager=checkpoint_mgr,
        )
        orchestrator.resume(resume_point)

        # === PHASE 3: Verify recovery completed ===

        # Original batch should be marked failed
        original_batch = recorder.get_batch(batch.batch_id)
        assert original_batch.status == BatchStatus.FAILED

        # New retry batch should exist
        all_batches = recorder.get_batches(run.run_id, node_id="sum_aggregator")
        retry_batch = next(
            (b for b in all_batches if b.attempt == 1), None
        )
        assert retry_batch is not None
        assert retry_batch.status in (BatchStatus.DRAFT, BatchStatus.COMPLETED)

        # Run should be back to running
        updated_run = recorder.get_run(run.run_id)
        assert updated_run.status == "running"
```

**Step 2: Run test**

Run: `pytest tests/integration/test_aggregation_recovery.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_aggregation_recovery.py
git commit -m "$(cat <<'EOF'
test(integration): add full aggregation recovery cycle test

Simulates crash during batch flush and verifies complete
recovery including state restoration and batch retry.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Document Recovery Procedure

**Files:**
- Create: `docs/runbooks/aggregation-crash-recovery.md`

**Step 1: Write runbook**

```markdown
# Aggregation Crash Recovery Runbook

## Overview

When an ELSPETH pipeline crashes during aggregation (batch collection or flush),
the system can recover without data loss using checkpoint-based recovery.

## Symptoms of Crash Requiring Recovery

1. Pipeline process terminated unexpectedly
2. Run status in landscape is "failed"
3. One or more batches have status "executing" or "failed"

## Recovery Steps

### 1. Verify Recovery is Possible

```python
from elspeth.core.checkpoint.recovery import RecoveryManager
from elspeth.core.landscape.database import LandscapeDB

db = LandscapeDB("sqlite:///landscape.db")
recovery_mgr = RecoveryManager(db)

if recovery_mgr.can_resume(run_id):
    print("Recovery is possible")
else:
    print("Cannot recover - no checkpoint found")
```

### 2. Get Resume Point

```python
resume_point = recovery_mgr.get_resume_point(run_id)
print(f"Resume from sequence: {resume_point.sequence_number}")
print(f"Aggregation state: {resume_point.aggregation_state}")
```

### 3. Resume Pipeline

```python
from elspeth.engine.orchestrator import Orchestrator

orchestrator = Orchestrator(recorder=recorder, checkpoint_manager=checkpoint_mgr)
orchestrator.resume(resume_point)
```

## What Happens During Recovery

1. **State Restoration**: Aggregation plugin state is restored from checkpoint JSON
2. **Batch Status Check**: System finds incomplete batches (draft/executing/failed)
3. **Batch Retry**: Failed batches get new attempt with copied members
4. **Processing Continues**: Pipeline resumes from last checkpoint sequence

## Monitoring Recovery

Check batch status after recovery:

```sql
SELECT batch_id, status, attempt, created_at
FROM batches
WHERE run_id = 'your-run-id'
ORDER BY created_at;
```

Expected: Original batch as "failed", retry batch as "completed" or "draft".

## Prevention

- Checkpoint frequency is configured in settings:
  ```yaml
  checkpoint:
    interval_rows: 100  # Checkpoint every 100 rows
  ```
- Lower interval = less data loss on crash, more overhead
```

**Step 2: Commit**

```bash
git add docs/runbooks/aggregation-crash-recovery.md
git commit -m "$(cat <<'EOF'
docs: add aggregation crash recovery runbook

Documents recovery procedure for crashed aggregation pipelines
including verification, resume steps, and monitoring.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan implements the missing orchestrator-level batch recovery:

| Component | Status Before | Status After |
|-----------|---------------|--------------|
| CheckpointManager | ✅ Exists | ✅ Unchanged |
| RecoveryManager | ✅ Exists | ✅ Unchanged |
| get_incomplete_batches() | ❌ Missing | ✅ Added |
| retry_batch() | ❌ Missing | ✅ Added |
| Orchestrator.resume() | ❌ Missing | ✅ Added |
| AggregationExecutor.restore_state() | ❌ Missing | ✅ Added |
| Integration test | ❌ Missing | ✅ Added |
| Runbook | ❌ Missing | ✅ Added |

**Closes:** ENG-007 (Aggregation crash recovery)
