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

### Step 3: Resume the Run

```bash
elspeth resume <RUN_ID>
```

The resume command:
1. Loads the original run's configuration from the audit trail
2. Finds the last valid checkpoint
3. Continues processing from that point
4. Records all events with the same run ID

### Step 4: Verify Completion

After the run completes:

```bash
# Check run status
sqlite3 runs/audit.db "SELECT status, completed_at FROM runs WHERE run_id = '<RUN_ID>';"

# Verify row counts
sqlite3 runs/audit.db "
  SELECT
    (SELECT COUNT(*) FROM rows WHERE run_id = '<RUN_ID>') as source_rows,
    (SELECT COUNT(*) FROM token_outcomes WHERE run_id = '<RUN_ID>' AND is_terminal = 1) as terminal_tokens;
"
```

Terminal rows should equal source rows (all rows reached a terminal state).

---

## Docker Resume

When running in Docker:

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  resume <RUN_ID>
```

**Important:** Mount the same `state/` directory that contains the original run's checkpoints.

---

## Troubleshooting

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

The resume command uses the configuration stored in the audit trail, not the current config file. If you need different settings, start a new run.

### "Duplicate row_id"

The checkpoint was corrupted or source data changed. Options:
1. Start a fresh run with `--force-new` (if available)
2. Manually mark the run as failed and start fresh

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
- [Configuration Reference](../reference/configuration.md#checkpoint-settings) - Checkpoint configuration
