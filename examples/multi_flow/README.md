# Multi-Flow Example

Demonstrates two independent flows in one pipeline run. Each named source has its own transform and sink; the flows share the run, audit database, scheduler, and final accounting, but they do not feed a shared queue or merge rows.

```text
signups source ─> normalize_signups ─> signup_events sink
tickets source ─> normalize_tickets ─> ticket_events sink
```

Run from the repository root:

```bash
elspeth run --settings examples/multi_flow/settings.yaml --execute
```

Expected result:

- `examples/multi_flow/output/signups.jsonl` contains two signup rows.
- `examples/multi_flow/output/tickets.jsonl` contains two support ticket rows.

Audit spot checks:

```bash
sqlite3 examples/multi_flow/runs/audit.db \
  "select source_name, source_node_id, lifecycle_state from run_sources order by source_name;"
sqlite3 examples/multi_flow/runs/audit.db \
  "select source_node_id, source_row_index, ingest_sequence from rows order by ingest_sequence;"
```
