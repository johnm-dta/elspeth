# Multi-Source Queue Example

Demonstrates two named source roots feeding the same queue before a shared transform and sink.

```text
orders source  ─┐
                ├─ inbound queue ─> normalize_rows ─> combined output
refunds source ─┘
```

This is v1 queue behavior: the queue is a durable scheduling and pass-through coordination point. It does not join or merge rows across sources.

Run from the repository root:

```bash
elspeth run --settings examples/multi_source_queue/settings.yaml --execute
```

Expected result: `examples/multi_source_queue/output/combined.jsonl` contains three rows, two from `orders.csv` and one from `refunds.csv`.

Audit spot checks:

```bash
sqlite3 examples/multi_source_queue/runs/audit.db \
  "select source_name, source_node_id, lifecycle_state from run_sources order by source_name;"
sqlite3 examples/multi_source_queue/runs/audit.db \
  "select source_node_id, source_row_index, ingest_sequence from rows order by ingest_sequence;"
```
