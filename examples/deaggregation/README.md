# Deaggregation (Batch Replicate) Example

Demonstrates 1-to-N row expansion using the `batch_replicate` aggregation plugin.

## What This Shows

A CSV source feeds rows into a `batch_replicate` aggregation that buffers 3 rows at a time. Each row is replicated N times based on its `copies` field, with a `copy_index` column added to distinguish copies.

```
source ─(deagg_in)─> [replicate_batch: batch_replicate, trigger=3] ─(output)─> replicated.csv
```

For example, a row with `copies: 3` produces 3 output rows with `copy_index` values 0, 1, 2.

## Running

```bash
elspeth run --settings examples/deaggregation/settings.yaml --execute
```

## Output

Results appear in `output/replicated.csv` with the original columns plus `copy_index`.

## Key Concepts

- **Deaggregation**: The inverse of aggregation — one input row becomes multiple output rows
- **Dynamic expansion**: The `copies` field controls how many copies each row produces
- **Batch triggering**: Rows are buffered until the count trigger fires (every 3 rows)
- **Token lineage**: Each expanded copy gets its own `token_id` with the parent token recorded for audit lineage
