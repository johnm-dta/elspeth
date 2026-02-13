# Batch Aggregation Example

Demonstrates the aggregation plugin with count-based triggers and group-by statistics.

## What This Shows

A CSV source feeds rows into a `batch_stats` aggregation that buffers 5 rows at a time, then computes sum, count, mean, and batch size grouped by `category`.

```
source ─(batch_in)─> [batch_totals: batch_stats, trigger=5] ─(output)─> CSV
```

## Running

```bash
elspeth run --settings examples/batch_aggregation/settings.yaml --execute
```

## Output

Results appear in `output/batch_summaries.csv` with columns: `count`, `sum`, `batch_size`, `mean`, `category`.

## Key Concepts

- **Aggregation triggers**: `count: 5` fires after every 5 rows
- **Group-by**: Statistics computed per `category` value within each batch
- **Output mode**: `transform` — the aggregation emits computed rows (not the original inputs)
