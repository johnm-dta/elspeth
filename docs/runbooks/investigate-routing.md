# Runbook: Investigate Routing

Explain why a row, token, or pipeline branch reached a specific destination.

Current as of 2026-05-20. This runbook uses the ADR-019 two-axis terminal
model: `token_outcomes.completed`, `token_outcomes.outcome`, and
`token_outcomes.path`.

---

## When To Use

- An auditor asks why a row was flagged, discarded, routed, or written.
- A row appears in an unexpected sink.
- A composed or hand-authored pipeline needs routing proof before rerun.
- You need to distinguish success, failure, and transient terminal paths.

## Inputs

- Landscape database path, usually from `settings.yaml` or the run environment.
- Run ID.
- Row ID or token ID.
- The settings file used for the run, when available.

## Preferred Procedure

### Step 1: Identify The Run

```bash
sqlite3 runs/audit.db "
  SELECT run_id, status, started_at, completed_at, config_hash
  FROM runs
  ORDER BY started_at DESC
  LIMIT 10;
"
```

If you are working from the Web UI, use the run ID shown in the run evidence or
diagnostics surface.

### Step 2: Find The Row Or Token

Find recent rows for a run:

```bash
sqlite3 runs/audit.db "
  SELECT row_id, row_index, source_data_hash
  FROM rows
  WHERE run_id = '<RUN_ID>'
  ORDER BY row_index
  LIMIT 20;
"
```

Find tokens for a row:

```bash
sqlite3 runs/audit.db "
  SELECT token_id, row_id, branch_name, created_at
  FROM tokens
  WHERE run_id = '<RUN_ID>'
    AND row_id = '<ROW_ID>'
  ORDER BY created_at, token_id;
"
```

### Step 3: Use `elspeth explain`

Use the CLI lineage read path before writing ad hoc SQL.

```bash
elspeth explain --run <RUN_ID> --row <ROW_ID> --database runs/audit.db
```

For non-interactive output:

```bash
elspeth explain --run <RUN_ID> --row <ROW_ID> --database runs/audit.db --no-tui
elspeth explain --run <RUN_ID> --token <TOKEN_ID> --database runs/audit.db --json
```

The explain output is the preferred evidence because it uses the maintained
Landscape query repositories and formatter.

### Step 4: Inspect Terminal Outcomes

Use direct SQL only when you need raw records for an incident note or audit
packet.

```bash
sqlite3 -header -column runs/audit.db "
  SELECT
    t.row_id,
    t.token_id,
    t.branch_name,
    o.completed,
    o.outcome,
    o.path,
    o.sink_name,
    o.batch_id,
    o.recorded_at
  FROM tokens t
  LEFT JOIN token_outcomes o
    ON o.run_id = t.run_id
   AND o.token_id = t.token_id
  WHERE t.run_id = '<RUN_ID>'
    AND t.row_id = '<ROW_ID>'
  ORDER BY t.created_at, o.recorded_at;
"
```

Interpretation:

- `completed = 1` means the token has a terminal outcome.
- `outcome = success` means the producer reported successful terminal handling.
- `outcome = failure` means the producer reported failed terminal handling.
- `outcome = transient` means the path is non-terminal or consumed by another
  producer, such as batch/coalesce handling.
- `path` explains the producer-declared terminal path.

### Step 5: Inspect Routing Events

```bash
sqlite3 -header -column runs/audit.db "
  SELECT
    ns.step_index,
    ns.node_id,
    n.plugin_name,
    e.label AS edge_label,
    re.mode,
    re.ordinal,
    re.reason_hash
  FROM routing_events re
  JOIN node_states ns
    ON ns.state_id = re.state_id
  JOIN nodes n
    ON n.run_id = ns.run_id
   AND n.node_id = ns.node_id
  JOIN edges e
    ON e.run_id = ns.run_id
   AND e.edge_id = re.edge_id
  JOIN tokens t
    ON t.run_id = ns.run_id
   AND t.token_id = ns.token_id
  WHERE ns.run_id = '<RUN_ID>'
    AND t.row_id = '<ROW_ID>'
  ORDER BY ns.step_index, re.ordinal, re.event_id;
"
```

Do not evaluate stored or configured expressions with Python `eval`. If a
condition needs source-level inspection, review the pipeline settings and the
production expression parser path instead.

### Step 6: Inspect Validation Or Transform Errors

Source validation errors:

```bash
sqlite3 -header -column runs/audit.db "
  SELECT node_id, row_id, destination, schema_mode, error, created_at
  FROM validation_errors
  WHERE run_id = '<RUN_ID>'
    AND (row_id = '<ROW_ID>' OR '<ROW_ID>' = '');
"
```

Transform errors for the row:

```bash
sqlite3 -header -column runs/audit.db "
  SELECT te.transform_id, te.token_id, te.destination, te.error_hash, te.created_at
  FROM transform_errors te
  JOIN tokens t
    ON t.run_id = te.run_id
   AND t.token_id = te.token_id
  WHERE te.run_id = '<RUN_ID>'
    AND t.row_id = '<ROW_ID>';
"
```

## Common Findings

| Symptom | Check |
|---------|-------|
| All rows took the same branch | Compare routing events across several rows and inspect the configured condition. |
| Row was discarded | Check validation and transform errors, then inspect `token_outcomes.path`. |
| Row entered a batch | Check `token_outcomes.batch_id` and batch records for the same run. |
| `explain` cannot find a row | Confirm the row belongs to the supplied run ID; row/run ownership is enforced. |
| Terminal count looks wrong | Use `completed`, not the retired `is_terminal` column. |

## See Also

- [Configuration Reference](../reference/configuration.md)
- [Incident Response](incident-response.md)
- [Database Maintenance](database-maintenance.md)
- [Architecture Overview](../../ARCHITECTURE.md)
