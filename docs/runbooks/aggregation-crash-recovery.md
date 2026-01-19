# Aggregation Crash Recovery Runbook

## Overview

When an ELSPETH pipeline crashes during aggregation (batch collection or flush),
the system can recover without data loss using checkpoint-based recovery.

## Symptoms of Crash Requiring Recovery

1. Pipeline process terminated unexpectedly
2. Run status in landscape is "failed"
3. Batches with status "executing" or "failed" exist

## Recovery Architecture

```
RecoveryManager                    Orchestrator
(determines IF/HOW)                (executes recovery)
        │                                  │
        │ can_resume(run_id)               │
        │ get_resume_point(run_id)         │
        │                                  │
        └──────────────────────────────────┤
                                           │
                                    resume(point, config, graph)
                                           │
                                    ┌──────┴──────┐
                                    │             │
                            Handle batches   Restore state
                                    │             │
                                    └──────┬──────┘
                                           │
                                    Continue processing
```

Key principle: **Orchestrator is stateless.** Each `resume()` call creates
fresh recorder and processor, just like `run()`. No hidden state.

## Recovery Steps

### 1. Verify Recovery is Possible

```python
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.landscape.database import LandscapeDB

db = LandscapeDB("sqlite:///landscape.db")
checkpoint_mgr = CheckpointManager(db)
recovery_mgr = RecoveryManager(db, checkpoint_mgr)

check = recovery_mgr.can_resume(run_id)
if check.can_resume:
    print("Recovery is possible")
else:
    print(f"Cannot recover: {check.reason}")
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

orchestrator = Orchestrator(db=db, checkpoint_manager=checkpoint_mgr)

# Use same config and graph as original run()
result = orchestrator.resume(resume_point, config, graph)
```

## What Happens During Recovery

1. **Batch Handling**:
   - EXECUTING batches → marked FAILED, then retried
   - FAILED batches → retried with attempt+1
   - DRAFT batches → continue normally

2. **State Restoration**:
   - Aggregation state from checkpoint restored to executor
   - Plugins can access via `get_restored_state(node_id)`

3. **Processing Continues**:
   - Unprocessed rows (after checkpoint) are reprocessed
   - Retry batches complete normally

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

Configure checkpoint frequency in settings:

```yaml
checkpoint:
  enabled: true
  frequency: every_n
  checkpoint_interval: 100  # Every 100 rows
```

Lower interval = less data loss on crash, more overhead.
