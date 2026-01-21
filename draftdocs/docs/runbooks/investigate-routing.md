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
# Search by field value in source payload
sqlite3 runs/audit.db "
  SELECT row_id, payload_hash
  FROM row_events
  WHERE run_id = '<RUN_ID>'
    AND event_type = 'SOURCE_EMIT'
    AND payload LIKE '%\"customer_id\": \"CUST123\"%'
  LIMIT 5;
"
```

If searching by a specific field value:

```bash
# For JSON payloads, use json_extract if available
sqlite3 runs/audit.db "
  SELECT row_id
  FROM row_events
  WHERE run_id = '<RUN_ID>'
    AND event_type = 'SOURCE_EMIT'
    AND json_extract(payload, '$.customer_id') = 'CUST123';
"
```

### Step 2: Use the Explain Command

Launch the lineage explorer TUI:

```bash
elspeth explain --run <RUN_ID> --row <ROW_ID>
```

The TUI shows:
- Source row and its content hash
- Each processing step (transforms, gates)
- Gate evaluation results and routing decisions
- Final destination and artifact hash

### Step 3: Manual Lineage Query (Alternative)

If you need raw data instead of the TUI:

```bash
# Get all events for a row
sqlite3 runs/audit.db "
  SELECT
    event_id,
    event_type,
    node_id,
    edge_label,
    payload_hash,
    created_at
  FROM row_events
  WHERE run_id = '<RUN_ID>'
    AND row_id = '<ROW_ID>'
  ORDER BY created_at;
"
```

### Step 4: Check Gate Evaluations

Find the gate decision that caused the routing:

```bash
sqlite3 runs/audit.db "
  SELECT
    node_id,
    edge_label,
    payload
  FROM row_events
  WHERE run_id = '<RUN_ID>'
    AND row_id = '<ROW_ID>'
    AND event_type = 'GATE_EVAL';
"
```

The `edge_label` shows which route was taken (`true`, `false`, or a custom label).

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
  ghcr.io/your-org/elspeth:latest \
  explain --run <RUN_ID> --row <ROW_ID>
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
# Find quarantine reason
sqlite3 runs/audit.db "
  SELECT payload
  FROM row_events
  WHERE run_id = '<RUN_ID>'
    AND row_id = '<ROW_ID>'
    AND event_type = 'QUARANTINE';
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
  SELECT event_type, node_id
  FROM row_events
  WHERE run_id = '<RUN_ID>'
    AND row_id = '<ROW_ID>'
  ORDER BY created_at;
"
```

Look for `CONSUMED_IN_BATCH` event type.

---

## Generating Audit Reports

For compliance reporting, export the complete lineage:

```bash
# Export all events for a specific row
sqlite3 -header -csv runs/audit.db "
  SELECT *
  FROM row_events
  WHERE run_id = '<RUN_ID>'
    AND row_id = '<ROW_ID>'
  ORDER BY created_at;
" > lineage_report.csv
```

---

## See Also

- [Configuration Reference](../reference/configuration.md#gate-settings) - Gate configuration
- [Incident Response](incident-response.md) - For broader investigations
