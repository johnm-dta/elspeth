# report_assemble example

Assemble a small text input into paginated markdown reports.

## What This Shows

A text source feeds rows into a `report_assemble` aggregation that buffers 2 rows at a time, then emits one report row per flush with pagination metadata.

```
source ─(lines)─> [pages: report_assemble, trigger=2] ─(output)─> JSON
```

## Running

```bash
elspeth run --settings examples/report_assemble/settings.yaml --execute
```

## Output

Results appear in `output/reports.json`. Each report row includes:

- `report_body` — the assembled text (markdown formatted with the configured title)
- `report_index` — 1-based page number
- `line_start`, `line_end` — inclusive source-row range covered by this page
- `line_count` — number of source rows in this page
- `lines_seen_total` — running total of source rows seen at flush time
- `flush_trigger` — `count` for full pages, `end_of_source` for the final partial page
- `is_end_of_source_report` — `True` only when this row is the final flush at end-of-source

With `trigger.count: 2` and 5 input lines, the run emits 3 reports: two full pages (count-triggered) and one final partial page (end-of-source-triggered).

## Key Concepts

- **Pagination metadata from batch context**: line ranges and flush trigger come from the real `AggregationBatchContext`, not an in-plugin counter.
- **End-of-source emission**: omit `trigger` to get a single whole-input report at end-of-source.
- **Output mode**: `transform` — the aggregation emits one assembled row per flush, replacing the source rows in the batch.
