# Checkpoint/Resume Example

Demonstrates crash recovery and graceful shutdown. The pipeline checkpoints its progress periodically, allowing interrupted runs to resume without reprocessing completed rows.

## What This Shows

A 20-row pipeline with checkpointing every 5 rows. If interrupted (Ctrl-C or crash), the run can be resumed from the last checkpoint.

```
source (20 rows) ─(validated)─> passthrough ─(processed)─> [status_gate] ─┬─ active.csv
                                                                           └─ other.csv
```

## Running

```bash
# Run the full pipeline
elspeth run --settings examples/checkpoint_resume/settings.yaml --execute
```

## Demonstrating Resume

### Step 1: Start a run and interrupt it

```bash
# Start the pipeline
elspeth run --settings examples/checkpoint_resume/settings.yaml --execute

# Press Ctrl-C while it's running to trigger graceful shutdown
# The pipeline will: flush buffers, write checkpoint, mark run INTERRUPTED
# Second Ctrl-C force-kills immediately (no checkpoint)
```

### Step 2: Check what happened

```bash
# The run is now INTERRUPTED with a checkpoint saved
# Look at the output files — only partially written
```

### Step 3: Resume from checkpoint

```bash
# Dry-run first — see what would happen
elspeth resume <run_id> --database examples/checkpoint_resume/runs/audit.db

# Execute the resume — processes only unfinished rows
elspeth resume <run_id> --database examples/checkpoint_resume/runs/audit.db --execute
```

The resume skips rows that were already persisted and appends new results to the output files.

If the source lifecycle is still `interrupted` and there are no persisted rows left
to replay, ELSPETH refuses to mark the run complete:

```json
{
  "event": "not_resumable",
  "reason": "source_not_exhausted"
}
```

That refusal is intentional. Current resume replays persisted row payloads through
the audit trail; it does not reopen the original source and prove that unread input
rows do not exist. Start a fresh run for that case, or use a future source-aware
resume path once implemented.

## Checkpoint Configuration

```yaml
checkpoint:
  enabled: true
  frequency: every_n          # Checkpoint every N rows
  checkpoint_interval: 5      # N = 5 rows
```

### Frequency Options

| Frequency | Trade-off |
|-----------|-----------|
| `every_row` | Safest for persisted-row replay. Higher I/O overhead. |
| `every_n` | Balanced — fewer checkpoint writes, but a mid-source interrupt may be refused if unread source rows could exist. |
| `aggregation_only` | Fastest — only checkpoint at aggregation boundaries. |

## Graceful Shutdown

When you press Ctrl-C during a run:

1. Signal handler sets a shutdown flag
2. Processing loop finishes the current row
3. Aggregation buffers are flushed
4. Pending sink writes complete
5. Checkpoint is created at current position
6. Run is marked `INTERRUPTED` (resumable)
7. CLI exits with code 3

A second Ctrl-C force-kills immediately (no checkpoint, run marked `FAILED`).

## Key Concepts

- **Checkpoints are stored in the Landscape database** — no separate checkpoint files
- **Topology validation on resume**: ELSPETH verifies the pipeline config hasn't changed
- **Sinks switch to append mode** on resume — no duplicate output
- **Idempotent**: Resume is safe to run multiple times
