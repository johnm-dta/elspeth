# Runbook: Resume Failed Run

Resume a pipeline that crashed or was interrupted.

---

## Symptoms

- Pipeline process terminated unexpectedly
- Run status shows `running` but process is not active
- Error message: "Run already in progress"

---

## Prerequisites

- Access to the audit database
- Access to the configuration file used for the original run
- The `state/` directory from the original run (contains checkpoints)

---

## Procedure

### Step 1: Identify the Failed Run

Find the run ID of the failed run:

```bash
# List recent runs and their status
sqlite3 runs/audit.db "
  SELECT run_id, status, started_at, completed_at,
         (SELECT COUNT(*) FROM rows WHERE rows.run_id = runs.run_id) as rows_processed
  FROM runs
  ORDER BY started_at DESC
  LIMIT 10;
"
```

Look for runs with status `running` that have no `completed_at` timestamp.

### Step 2: Check Checkpoint State

Verify checkpoints exist for the run:

```bash
sqlite3 runs/audit.db "
  SELECT checkpoint_id, row_id, token_id, created_at
  FROM checkpoints
  WHERE run_id = '<RUN_ID>'
  ORDER BY created_at DESC
  LIMIT 5;
"
```

If no checkpoints exist, the run cannot be resumed - you must start fresh.

### Step 3: Check Resume Compatibility

```bash
elspeth resume <RUN_ID>
```

Without `--execute`, the resume command checks whether the run can be resumed,
reports the resume point, and validates checkpoint compatibility. To actually
continue processing:

```bash
elspeth resume <RUN_ID> --execute
```

The resume command:
1. Loads the settings needed to validate checkpoint compatibility
2. Finds the last valid checkpoint
3. Continues processing from that point when `--execute` is present
4. Records all events with the same run ID

### Step 4: Verify Completion

After the run completes:

```bash
# Check run status
sqlite3 runs/audit.db "SELECT status, completed_at FROM runs WHERE run_id = '<RUN_ID>';"

# Verify row and terminal-token counts
sqlite3 runs/audit.db "
  SELECT
    (SELECT COUNT(*) FROM rows WHERE run_id = '<RUN_ID>') as source_rows,
    (SELECT COUNT(*) FROM token_outcomes WHERE run_id = '<RUN_ID>' AND completed = 1) as terminal_tokens;
"
```

In linear pipelines the terminal-token count usually matches source rows. In
fork, expand, batch, or coalesce pipelines, use `elspeth explain` for a sample
row and confirm every live token reached an expected terminal path.

---

## Docker Resume

When running in Docker:

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:latest \
  resume <RUN_ID> --execute
```

**Important:** Mount the same `state/` directory that contains the original run's checkpoints.

---

## Troubleshooting

### Pre-2026-01-24 Checkpoints Are Invalid

All checkpoints created before 2026-01-24 are invalid due to node ID format changes introduced in the routing refactor. Attempting to resume from a pre-2026-01-24 checkpoint will fail. Delete old checkpoint files and re-run affected pipelines.

The useful migration rule from the old RC-2 checkpoint note is:

1. Checkpoints from the old `token_ids`-only format cannot be restored.
2. Delete old checkpoint files for the affected run.
3. Re-run the pipeline so new checkpoints store full token metadata.

For full historical context see the [RC-2 Checkpoint Fix Post-Mortem](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc-2-checkpoint-fix-postmortem.md) in the 2026-05-19 docs archive.

### "Run not found"

The run ID doesn't exist in the audit database:

```bash
# List all run IDs
sqlite3 runs/audit.db "SELECT run_id FROM runs;"
```

### "No checkpoint available"

The run crashed before creating any checkpoints. Start a new run instead:

```bash
elspeth run --settings pipeline.yaml --execute
```

### "Configuration mismatch"

The resume command validates the checkpoint against the settings it loads for
the resume attempt. If you need different settings, start a new run instead of
resuming the old one.

### "Duplicate row_id"

The checkpoint was corrupted or source data changed. Options:
1. Start a fresh run.
2. Preserve the failed run's audit database and checkpoint files if they are
   needed for incident evidence.

---

## Prevention

To reduce resume scenarios:

1. **Use frequent checkpoints** for critical pipelines:
   ```yaml
   checkpoint:
     enabled: true
     frequency: every_row
   ```

2. **Monitor pipeline processes** with health checks

3. **Use Docker with restart policies**:
   ```yaml
   services:
     elspeth:
       restart: on-failure:3
   ```

---

## See Also

- [Incident Response](incident-response.md) - For investigating root cause
- [Scheduler Lease Recovery](scheduler-lease-recovery.md) - For diagnosing stuck `leased` work items, SCREAM invariants, and lease-expiry churn before invoking `elspeth resume`
- [Configuration Reference](../reference/configuration.md#checkpoint-settings) - Checkpoint configuration
