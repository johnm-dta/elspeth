# concurrent_scheduler

This example proves that the ELSPETH 0.6.0 scheduler keeps multiple token
lifecycles open *simultaneously*. Two CSV sources each contribute 3 rows to a
shared queue; a `batch_stats` aggregation with a count-6 trigger can only fire
if all six tokens are alive at once.  Successful completion is the proof.

## What This Shows

The 0.6.0 concurrent token scheduler (ADR-026) maintains an in-process work
queue that holds tokens from multiple sources open concurrently rather than
draining one source at a time. A `batch_stats` count-6 barrier requires exactly
six items to be present before it will flush: under one-at-a-time draining the
barrier would never accumulate six members and the run would deadlock. The fact
that the run completes with a single aggregate row over all six tokens is the
direct proof of concurrent scheduling.

```
stream_a.csv (3 rows) ─┐
                       ├─> [queue: merged] ─> (batch_stats count=6) ─> rendezvous.csv
stream_b.csv (3 rows) ─┘
```

## Running

```bash
source .venv/bin/activate
elspeth run --settings examples/concurrent_scheduler/settings.yaml --execute
```

No server required. The example is pure-data and self-contained.

## Output

`output/rendezvous.csv` — one aggregate row:

| count | sum   | batch_size | mean |
|-------|-------|------------|------|
| 6     | 210.0 | 6          | 35.0 |

`count=6` and `batch_size=6` are the load-bearing proof that all six tokens
were alive concurrently when the barrier fired. `sum=210` (10+20+30+40+50+60)
is only possible if both sources' tokens entered the same barrier fire.

## Verifying Both Sources Contributed

After a run, query the read-only audit DB to confirm that the aggregation drew
from both named sources. Source attribution lives on `run_sources`, not on
`rows` (there is no `rows.source_name` column — use `run_sources.source_name`).

```bash
# Get the run_id from the most recent run:
RUN_ID=$(sqlite3 "file:examples/concurrent_scheduler/runs/audit.db?mode=ro" \
  "PRAGMA query_only=ON; SELECT run_id FROM runs LIMIT 1;")

# Confirm both sources are registered for this run:
sqlite3 "file:examples/concurrent_scheduler/runs/audit.db?mode=ro" \
  "PRAGMA query_only=ON; SELECT DISTINCT source_name FROM run_sources WHERE run_id='$RUN_ID';"
```

Expected output:
```
stream_a
stream_b
```

For the fuller cross-source attribution proof (which rows actually entered the
barrier, joined back to their source):

```bash
sqlite3 "file:examples/concurrent_scheduler/runs/audit.db?mode=ro" \
  "PRAGMA query_only=ON; SELECT DISTINCT rs.source_name FROM rows r \
   JOIN run_sources rs ON rs.source_node_id = r.source_node_id \
   AND rs.run_id = r.run_id;"
```

Expected: both `stream_a` and `stream_b` (6 source rows total).

## Key Concepts

- **Concurrent token scheduling (ADR-026):** The scheduler holds multiple token
  lifecycles open simultaneously; all six tokens from both sources are active in
  the work queue before any barrier flush can occur.
- **`batch_stats` count trigger:** The `trigger: {count: 6}` setting means the
  aggregation only fires when exactly 6 items have accumulated — a rendezvous
  barrier. `value_field: amount` names the numeric field to aggregate;
  `compute_mean: true` adds `mean` to the output.
- **Multi-source fan-in via a named queue:** Both `stream_a` and `stream_b`
  declare `on_success: merged`; the explicit `queues: {merged: {}}` node in the
  settings makes multi-producer fan-in legal and durable.
- **`output_mode: transform`:** The `batch_stats` aggregation emits its
  aggregate row downstream as a transform output (not a sink write), which is
  then routed to the `output` CSV sink.
