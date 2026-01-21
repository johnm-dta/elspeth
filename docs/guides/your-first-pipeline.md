# Your First Pipeline

Build and run an ELSPETH pipeline in 10 minutes. No external APIs required.

---

## What You'll Build

A transaction routing pipeline that:
1. **Reads** transaction data from a CSV file
2. **Routes** high-value transactions (amount > $1000) to a separate file
3. **Records** every routing decision in an audit trail

```
input.csv → [threshold gate] → normal.csv (amount ≤ 1000)
                    ↓
            high_values.csv (amount > 1000)
```

By the end, you'll be able to run `elspeth explain --run latest --row 2` and explore why Bob's $1500 transaction was flagged as high-value.

---

## Prerequisites

Choose your environment:

| Environment | Requirements |
|-------------|--------------|
| **Local Python** | Python 3.11+, uv package manager |
| **Docker** | Docker installed and running |

---

## Option A: Running Locally (Python)

### Step 1: Install ELSPETH

```bash
# Clone the repository
git clone https://github.com/johnm-dta/elspeth.git
cd elspeth

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify installation
elspeth --version
```

### Step 2: Explore the Example

The example is already set up in `examples/threshold_gate/`. Let's look at what's there:

```bash
ls examples/threshold_gate/
```

```
input.csv       # Source data
settings.yaml   # Pipeline configuration
output/         # Where results go
runs/           # Audit trail storage
```

**Input data** (`examples/threshold_gate/input.csv`):

```csv
id,name,amount,category
1,Alice,500,retail
2,Bob,1500,wholesale
3,Charlie,250,retail
4,Diana,3000,wholesale
5,Eve,750,retail
6,Frank,2000,corporate
7,Grace,100,retail
8,Henry,5000,corporate
```

**Expected routing:**
- Alice, Charlie, Eve, Grace → `normal.csv` (amount ≤ 1000)
- Bob, Diana, Frank, Henry → `high_values.csv` (amount > 1000)

### Step 3: Understand the Configuration

Open `examples/threshold_gate/settings.yaml`:

```yaml
# SENSE: Where data comes from
datasource:
  plugin: csv
  options:
    path: examples/threshold_gate/input.csv
    schema:
      mode: free
      fields:
        - "id: int"
        - "amount: int"
    on_validation_failure: discard

# DECIDE: How to route rows
gates:
  - name: amount_threshold
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values    # High amounts → high_values sink
      "false": continue      # Normal amounts → output_sink

# ACT: Where data goes
sinks:
  output:
    plugin: csv
    options:
      path: examples/threshold_gate/output/normal.csv
      schema:
        fields: dynamic
  high_values:
    plugin: csv
    options:
      path: examples/threshold_gate/output/high_values.csv
      schema:
        fields: dynamic

output_sink: output

# Audit trail
landscape:
  url: sqlite:///examples/threshold_gate/runs/audit.db
```

**Key concepts:**

| Section | Purpose |
|---------|---------|
| `datasource` | **SENSE** - Load data from CSV, validate schema |
| `gates` | **DECIDE** - Route based on condition |
| `sinks` | **ACT** - Write to output files |
| `landscape` | **AUDIT** - Record everything |

### Step 4: Run the Pipeline

```bash
# Validate configuration first
elspeth validate --settings examples/threshold_gate/settings.yaml

# Execute the pipeline
elspeth run --settings examples/threshold_gate/settings.yaml --execute
```

**Expected output:**

```
Run abc123 completed successfully
  Rows processed: 8
  Normal transactions: 4
  High-value transactions: 4
  Audit trail: examples/threshold_gate/runs/audit.db
```

### Step 5: Check the Results

```bash
# Normal transactions (≤ $1000)
cat examples/threshold_gate/output/normal.csv
```

```csv
id,name,amount,category
1,Alice,500,retail
3,Charlie,250,retail
5,Eve,750,retail
7,Grace,100,retail
```

```bash
# High-value transactions (> $1000)
cat examples/threshold_gate/output/high_values.csv
```

```csv
id,name,amount,category
2,Bob,1500,wholesale
4,Diana,3000,wholesale
6,Frank,2000,corporate
8,Henry,5000,corporate
```

### Step 6: Explain a Decision

This is where ELSPETH shines. Ask "why did row 2 (Bob) get routed to high_values?"

```bash
# Launch the lineage explorer TUI
elspeth explain --run latest --row 2
```

This launches an interactive terminal UI where you can explore:
- The source row and its content hash
- Each processing step (transforms, gates)
- Gate evaluation results and routing decisions
- Final destination and artifact hash

> **Note:** Text output via `--no-tui` is planned for a future release. Currently, the TUI provides the lineage exploration interface.

Every decision is traceable. If an auditor asks "why was this transaction flagged?", you have the answer.

---

## Option B: Running with Docker

### Step 1: Set Up Directory Structure

Create a working directory with the required structure:

```bash
mkdir -p my-pipeline/{config,input,output,state}
cd my-pipeline
```

### Step 2: Create Input Data

```bash
cat > input/transactions.csv << 'EOF'
id,name,amount,category
1,Alice,500,retail
2,Bob,1500,wholesale
3,Charlie,250,retail
4,Diana,3000,wholesale
5,Eve,750,retail
6,Frank,2000,corporate
7,Grace,100,retail
8,Henry,5000,corporate
EOF
```

### Step 3: Create Pipeline Configuration

```bash
cat > config/pipeline.yaml << 'EOF'
# SENSE: Load from CSV
datasource:
  plugin: csv
  options:
    path: /app/input/transactions.csv  # Container path!
    schema:
      mode: free
      fields:
        - "id: int"
        - "amount: int"
    on_validation_failure: discard

# DECIDE: Route high-value transactions
gates:
  - name: amount_threshold
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": continue

# ACT: Write to output files
sinks:
  output:
    plugin: csv
    options:
      path: /app/output/normal.csv  # Container path!
      schema:
        fields: dynamic
  high_values:
    plugin: csv
    options:
      path: /app/output/high_values.csv  # Container path!
      schema:
        fields: dynamic

output_sink: output

# Audit trail
landscape:
  url: sqlite:////app/state/audit.db  # Note: 4 slashes for absolute path
EOF
```

**Important:** Use container paths (`/app/input/...`), not host paths (`./input/...`).

### Step 4: Validate the Configuration

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  ghcr.io/johnm-dta/elspeth:latest \
  validate --settings /app/config/pipeline.yaml
```

**Expected output:**

```
Configuration valid: /app/config/pipeline.yaml
  Source: csv
  Transforms: 0
  Gates: 1 (amount_threshold)
  Sinks: 2 (output, high_values)
```

### Step 5: Run the Pipeline

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:latest \
  run --settings /app/config/pipeline.yaml --execute
```

**Expected output:**

```
Run abc123 completed successfully
  Rows processed: 8
  Normal transactions: 4
  High-value transactions: 4
```

### Step 6: Check the Results

```bash
# Normal transactions
cat output/normal.csv

# High-value transactions
cat output/high_values.csv
```

### Step 7: Explain a Decision

For Docker environments where TUI isn't available, query the audit database directly:

```bash
# Query routing decision for row 2
docker run --rm \
  -v $(pwd)/state:/app/state:ro \
  --entrypoint sqlite3 \
  ghcr.io/johnm-dta/elspeth:latest \
  /app/state/audit.db \
  "SELECT t.token_id, ns.node_id, ns.status, ns.input_hash
   FROM tokens t
   JOIN rows r ON t.row_id = r.row_id
   JOIN node_states ns ON t.token_id = ns.token_id
   WHERE r.row_index = 2
   ORDER BY ns.step_index;"
```

> **Note:** Text output via `--no-tui` is planned for a future release. Currently, use the TUI for interactive lineage exploration or query the audit database directly for CI/CD environments.

---

## Using docker-compose

For repeated runs, docker-compose is more convenient:

```yaml
# docker-compose.yaml
services:
  elspeth:
    image: ghcr.io/johnm-dta/elspeth:latest
    volumes:
      - ./config:/app/config:ro
      - ./input:/app/input:ro
      - ./output:/app/output
      - ./state:/app/state
```

```bash
# Validate
docker compose run --rm elspeth validate --settings /app/config/pipeline.yaml

# Run
docker compose run --rm elspeth run --settings /app/config/pipeline.yaml --execute

# Explain (interactive TUI)
docker compose run -it --rm elspeth explain --run latest --row 2
```

---

## What Just Happened?

Let's trace through the pipeline:

### 1. SENSE (Source)

The CSV source:
- Loaded 8 rows from `input.csv`
- Validated each row against the schema (`id: int`, `amount: int`)
- Coerced string values to integers (CSV stores everything as strings)
- Would have routed invalid rows to `on_validation_failure` sink (we used `discard`)

### 2. DECIDE (Gate)

The threshold gate:
- Evaluated `row['amount'] > 1000` for each row
- Rows with `true` → routed to `high_values` sink
- Rows with `false` → continued to default `output` sink

### 3. ACT (Sinks)

Two CSV sinks:
- `output` → wrote 4 normal transactions
- `high_values` → wrote 4 high-value transactions
- Each sink computed a content hash for the audit trail

### 4. AUDIT (Landscape)

The audit database recorded:
- Run configuration (so you can see exactly what settings were used)
- Every row's journey through the pipeline
- Every gate evaluation with the condition result
- Content hashes of all output files

---

## Try These Modifications

### Change the Threshold

Edit the gate condition:

```yaml
gates:
  - name: amount_threshold
    condition: "row['amount'] > 500"  # Lower threshold
```

Re-run and see how the routing changes.

### Add a Third Tier

Route "premium" transactions (> $2500) separately:

```yaml
gates:
  - name: premium_check
    condition: "row['amount'] > 2500"
    routes:
      "true": premium
      "false": continue

  - name: high_value_check
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": continue

sinks:
  output:
    # ... normal transactions
  high_values:
    # ... high-value transactions
  premium:
    plugin: csv
    options:
      path: /app/output/premium.csv
      schema:
        fields: dynamic
```

### Add a Transform

Add a field before routing:

```yaml
row_plugins:
  - plugin: field_mapper
    options:
      schema:
        fields: dynamic
      computed:
        tier: "row['amount'] > 2500 and 'premium' or (row['amount'] > 1000 and 'high' or 'normal')"

gates:
  - name: tier_router
    condition: "row['tier'] == 'premium'"
    routes:
      "true": premium
      "false": continue
```

---

## Troubleshooting

### "File not found" in Docker

**Symptom:** `FileNotFoundError: /app/input/transactions.csv`

**Fix:** Check that:
1. Volume is mounted: `-v $(pwd)/input:/app/input:ro`
2. Config uses container paths: `path: /app/input/...` (not `./input/...`)

### "Permission denied" on output

**Symptom:** `PermissionError` when writing to `/app/output/`

**Fix:** Ensure output directory exists and is writable:
```bash
mkdir -p output
chmod 777 output
```

### "Invalid schema" error

**Symptom:** `ValidationError: field 'amount' expected int, got str`

**Fix:** Add type coercion in schema:
```yaml
schema:
  mode: free
  fields:
    - "amount: int"  # Will coerce "1500" to 1500
```

### Nothing in high_values.csv

**Symptom:** All rows go to normal.csv

**Fix:** Ensure the source schema coerces numeric fields. The expression parser does NOT allow function calls like `int()`:
```yaml
# In datasource config - coerce to int at source
datasource:
  plugin: csv
  options:
    schema:
      mode: free
      fields:
        - "amount: int"  # Coerces "1500" to 1500

# Then in gate condition - amount is already an int
condition: "row['amount'] > 1000"
```

---

## Next Steps

Now that you've built your first pipeline:

1. **Add an LLM transform** - See `examples/openrouter_sentiment/` for LLM classification
2. **Export the audit trail** - Add `landscape.export` to create signed exports
3. **Build a custom plugin** - See [PLUGIN.md](../../PLUGIN.md) for plugin development
4. **Explore the architecture** - See [ARCHITECTURE.md](../../ARCHITECTURE.md) for system design

---

## Quick Reference

### Local Commands

```bash
elspeth validate --settings path/to/settings.yaml
elspeth run --settings path/to/settings.yaml --execute
elspeth explain --run latest --row <row_id>
elspeth plugins list
```

### Docker Commands

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:latest \
  <command>
```

| Command | Description |
|---------|-------------|
| `validate --settings /app/config/pipeline.yaml` | Check configuration |
| `run --settings /app/config/pipeline.yaml --execute` | Run pipeline |
| `explain --run latest --row N` | Explain decision (TUI) |
| `plugins list` | List available plugins |
| `--help` | Show all commands |

---

## See Also

- [README.md](../../README.md) - Project overview
- [Docker Guide](docker.md) - Complete Docker deployment
- [PLUGIN.md](../../PLUGIN.md) - Creating custom plugins
- [examples/](../../../examples/) - More example pipelines
