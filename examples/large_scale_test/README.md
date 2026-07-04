# Large-Scale Test Example

This example demonstrates ELSPETH's durable scheduler and audit trail with large datasets.
It is a throughput and audit-overhead example, not a quick smoke test.

## What This Example Shows

- **Scale Testing**: Process tens of thousands of rows with full audit trail
- **Gate Routing**: Route high-value transactions to a separate sink based on value threshold
- **Durable Scheduler Cost**: Measure throughput with recoverable work items, scheduler events, and full audit writes
- **Lineage**: Explore complete lineage for any row in a large dataset

## Dataset

Procedurally generated CSV with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Sequential row identifier (1, 2, 3...) |
| `value` | float | Random value between 0-10000 |
| `category` | str | One of 5 categories (A, B, C, D, E) |
| `priority` | int | Priority level 1-5 |
| `timestamp` | str | ISO 8601 timestamp (increments by second) |

## Pipeline Flow

```
CSV Source (50k rows)
    ↓
Value Threshold Gate
    ├─→ value >= 5000 → high_value.csv (~10k rows)
    └─→ value < 5000  → normal.csv (~40k rows)
```

## Usage

### 1. Generate Test Data

Generate 50,000 rows (default):

```bash
python examples/large_scale_test/generate_data.py
```

Generate custom row count:

```bash
# 1,000 rows (quick local check)
python examples/large_scale_test/generate_data.py 1000

# 10,000 rows
python examples/large_scale_test/generate_data.py 10000

# 100,000 rows (stress test)
python examples/large_scale_test/generate_data.py 100000
```

### 2. Run Pipeline

```bash
elspeth run --settings examples/large_scale_test/settings.yaml --execute
```

### 3. Explore Results

Check output files:

```bash
# Count lines in each output
wc -l examples/large_scale_test/output/*.csv

# View high-value transactions
head examples/large_scale_test/output/high_value.csv
```

Explore lineage for any row:

```bash
# Pick any row ID from the dataset
elspeth explain --run latest --row 1234 --database examples/large_scale_test/runs/audit.db
```

Query the audit database directly:

```bash
sqlite3 examples/large_scale_test/runs/audit.db

# Check row counts
SELECT state, COUNT(*) FROM tokens GROUP BY state;

# View gate routing decisions
SELECT * FROM routing_events LIMIT 10;
```

## Checkpoint Policy

This example declares a coarse checkpoint policy explicitly:

```yaml
checkpoint:
  enabled: true
  frequency: every_n
  checkpoint_interval: 100
```

That setting is the intended durability/performance balance for this high-cardinality
local benchmark. When comparing `every_row`, `every_n`, `aggregation_only`, or
disabled checkpointing, record the checkpoint policy with the timing results; do
not rely on runtime defaults.

## Performance Expectations

The current durable scheduler mode records recoverable scheduler state and
per-transition scheduler events for every row. That makes this example much
heavier than the older pre-durable-scheduler benchmark numbers.

Measured local SQLite evidence from the multi-source-token-scheduler worktree:
5,000 generated rows completed in about 50 seconds with all rows successful and
matching scheduler event counts.

Use these as order-of-magnitude expectations on a local SQLite database:

| Row Count | Processing Time | Throughput |
|-----------|----------------|------------|
| 1,000 | ~10 seconds | ~100 rows/sec |
| 5,000 | ~50 seconds | ~100 rows/sec |
| 50,000 | ~7-9 minutes | ~100 rows/sec |

Actual performance depends on hardware, disk I/O, the declared checkpoint
policy, payload storage, and database backend. PostgreSQL or a future
high-cardinality scheduler batching mode may have different characteristics.

## What Gets Audited

For each row, ELSPETH records:

- Source row entry with content hash
- Initial token and token terminal outcome
- Source and gate node states
- Routing decision for the gate branch
- Durable scheduler work item transitions
- Sink node state and output artifact hash

The durable scheduler path writes more than five rows of audit data per input
row. A 50k-row run should be treated as a long-running local benchmark.

## Use Cases

This example is useful for:

1. **Performance Testing**: Measure ELSPETH durable-audit throughput with realistic data volumes
2. **Audit Verification**: Verify complete lineage at scale
3. **Load Testing**: Stress test with 100k+ rows
4. **Development**: Test plugins with large datasets before production
5. **Benchmarking**: Compare performance across checkpoint and database configurations

## Extending This Example

### Add Transforms

```yaml
transforms:
  - name: your_transform_0
    plugin: your_transform
    input: gate_in
    on_success: output
    on_error: output
    options:
      schema:
        mode: observed
      # transform-specific options
```

### Multiple Gates

```yaml
gates:
  - name: value_threshold
    condition: "row['value'] >= 5000"
    routes:
      "true": high_value
      "false": continue

  - name: category_filter
    condition: "row['category'] in ['A', 'B']"
    routes:
      "true": priority_categories
      "false": continue
```

### Add Aggregation

See the `batch_aggregation` example for batch processing at scale.

## Cleanup

Remove generated data and outputs:

```bash
rm examples/large_scale_test/input.csv
rm -rf examples/large_scale_test/output/*.csv
rm -rf examples/large_scale_test/runs/*.db
```
