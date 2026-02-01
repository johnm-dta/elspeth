# Large-Scale Test Example

This example demonstrates ELSPETH's performance and auditability with large datasets (10k-100k rows).

## What This Example Shows

- **Scale Testing**: Process tens of thousands of rows with full audit trail
- **Gate Routing**: Route high-value transactions to a separate sink based on value threshold
- **Performance**: Measure throughput and audit overhead at scale
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

Generate custom row count (10k-100k recommended):

```bash
# 10,000 rows (fast)
python examples/large_scale_test/generate_data.py 10000

# 100,000 rows (stress test)
python examples/large_scale_test/generate_data.py 100000
```

### 2. Run Pipeline

```bash
uv run elspeth run -s examples/large_scale_test/settings.yaml --execute
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

## Performance Expectations

Typical throughput on modern hardware:

| Row Count | Processing Time | Throughput |
|-----------|----------------|------------|
| 10,000 | ~1-2 seconds | ~5,000-10,000 rows/sec |
| 50,000 | ~5-10 seconds | ~5,000-10,000 rows/sec |
| 100,000 | ~10-20 seconds | ~5,000-10,000 rows/sec |

*Note: Actual performance depends on hardware, disk I/O, and database backend (SQLite vs PostgreSQL).*

## What Gets Audited

For each of the 50k rows, ELSPETH records:

- ✅ Source row entry with content hash
- ✅ Transform input/output hashes
- ✅ Gate evaluation result (`true`/`false`)
- ✅ Routing decision (which sink)
- ✅ Terminal state (`COMPLETED` or `ROUTED`)
- ✅ Output artifact hash

**Total audit records**: ~250k entries for 50k rows (5 records per row)

## Use Cases

This example is useful for:

1. **Performance Testing**: Measure ELSPETH throughput with realistic data volumes
2. **Audit Verification**: Verify complete lineage at scale
3. **Load Testing**: Stress test with 100k+ rows
4. **Development**: Test plugins with large datasets before production
5. **Benchmarking**: Compare performance across different configurations

## Extending This Example

### Add Transforms

```yaml
row_plugins:
  - plugin: your_transform
    options:
      schema:
        fields: dynamic
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
