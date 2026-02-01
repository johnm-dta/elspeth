# Runbook: Investigate Routing

Explain why a specific row was routed to a particular destination.

---

## Symptoms

- Auditor asks "why was transaction X flagged?"
- Row appeared in unexpected sink
- Need to verify routing logic is working correctly

---

## Prerequisites

- Run ID where the row was processed
- Row identifier (row_id or content from the row)
- Access to the audit database

---

## Procedure

### Step 1: Find the Row

If you have the row content but not the row_id:

```bash
# Search by row index
sqlite3 runs/audit.db "
  SELECT row_id, source_data_hash
  FROM rows
  WHERE run_id = '<RUN_ID>'
  ORDER BY row_index
  LIMIT 10;
"
```

To find a specific row, you'll need to check the source data. If you know the row index:

```bash
# Find by row index
sqlite3 runs/audit.db "
  SELECT row_id, row_index, source_data_hash
  FROM rows
  WHERE run_id = '<RUN_ID>'
    AND row_index = 42;
"
```

### Step 2: Use the Explain Command

Launch the lineage explorer TUI:

```bash
elspeth explain --run <RUN_ID> --row <ROW_ID> --database <path/to/audit.db>
```

The TUI shows:
- Source row and its content hash
- Each processing step (transforms, gates)
- Gate evaluation results and routing decisions
- Final destination and artifact hash

### Step 3: Manual Lineage Query (Alternative)

If you need raw data instead of the TUI:

```bash
# Get all processing states for a row
sqlite3 runs/audit.db "
  SELECT
    ns.state_id,
    ns.node_id,
    n.plugin_name,
    ns.status,
    ns.input_hash,
    ns.output_hash,
    ns.started_at
  FROM node_states ns
  JOIN tokens t ON ns.token_id = t.token_id
  JOIN nodes n ON ns.node_id = n.node_id
  WHERE t.row_id = '<ROW_ID>'
  ORDER BY ns.step_index;
"
```

### Step 4: Check Gate Evaluations

Find the gate decision that caused the routing:

```bash
sqlite3 runs/audit.db "
  SELECT
    ns.node_id,
    n.plugin_name,
    re.mode,
    e.label as route_taken
  FROM routing_events re
  JOIN node_states ns ON re.state_id = ns.state_id
  JOIN edges e ON re.edge_id = e.edge_id
  JOIN nodes n ON ns.node_id = n.node_id
  JOIN tokens t ON ns.token_id = t.token_id
  WHERE t.row_id = '<ROW_ID>';
"
```

The `label` shows which route was taken (`true`, `false`, or a named sink).

### Step 5: Verify the Condition

Check the gate configuration that was used:

```bash
sqlite3 runs/audit.db "
  SELECT config
  FROM runs
  WHERE run_id = '<RUN_ID>';
"
```

Parse the JSON config to find the gate condition, then manually evaluate:

```python
# Example: verify the condition
row = {"amount": 1500}  # From the payload
condition = "row['amount'] > 1000"  # From the config
print(eval(condition))  # True
```

---

## Docker Usage

```bash
docker run --rm \
  -v $(pwd)/state:/app/state:ro \
  ghcr.io/johnm-dta/elspeth:latest \
  explain --run <RUN_ID> --row <ROW_ID> --database <path/to/audit.db>
```

---

## Common Scenarios

### Row Went to Wrong Sink

1. Find the gate that made the decision
2. Check the condition in the config
3. Verify the row's field values at that point
4. Check if any transforms modified the field before the gate

### Row Was Quarantined

```bash
# Find validation errors (source quarantine)
sqlite3 runs/audit.db "
  SELECT error, schema_mode, destination
  FROM validation_errors
  WHERE run_id = '<RUN_ID>';
"

# Find transform errors
sqlite3 runs/audit.db "
  SELECT te.transform_id, te.error_details_json, te.destination
  FROM transform_errors te
  JOIN tokens t ON te.token_id = t.token_id
  WHERE t.row_id = '<ROW_ID>';
"
```

Common reasons:
- Schema validation failure
- Transform error
- Missing required field

### Row Disappeared

Check if it was consumed by an aggregation:

```bash
sqlite3 runs/audit.db "
  SELECT to.outcome, to.sink_name, to.batch_id
  FROM token_outcomes to
  JOIN tokens t ON to.token_id = t.token_id
  WHERE t.row_id = '<ROW_ID>';
"
```

Look for outcome `consumed_in_batch` - this means the row was aggregated into a batch.

---

## Generating Audit Reports

For compliance reporting, export the complete lineage:

```bash
# Export complete lineage for a specific row
sqlite3 -header -csv runs/audit.db "
  SELECT
    r.row_id, r.row_index, r.source_data_hash,
    t.token_id, t.branch_name,
    ns.node_id, ns.step_index, ns.status, ns.input_hash, ns.output_hash,
    to.outcome, to.sink_name
  FROM rows r
  JOIN tokens t ON r.row_id = t.row_id
  LEFT JOIN node_states ns ON t.token_id = ns.token_id
  LEFT JOIN token_outcomes to ON t.token_id = to.token_id
  WHERE r.row_id = '<ROW_ID>'
  ORDER BY ns.step_index;
" > lineage_report.csv
```

---

## See Also

- [Configuration Reference](../reference/configuration.md#gate-settings) - Gate configuration
- [Incident Response](incident-response.md) - For broader investigations
