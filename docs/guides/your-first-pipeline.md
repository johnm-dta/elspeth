# Your First Pipeline

Build and run an ELSPETH pipeline from the CLI, the browser, or Docker.
The CLI and Docker paths require no external APIs. The Web UI path uses the
same runtime and audit model, and uses the Web Composer when you want browser
authoring and execution.

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

By the end, you'll have a complete audit trail showing why Bob's $1500 transaction was routed to `high_values.csv`.

---

## Prerequisites

Choose your environment:

| Environment | Requirements |
|-------------|--------------|
| **Local Python** | Python 3.11+, uv package manager |
| **Web UI** | Python 3.11+, uv, npm, `.[webui]` extra, local auth user |
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
source:
  plugin: csv
  on_success: gate_in               # Route validated rows to the gate
  options:
    path: examples/threshold_gate/input.csv
    schema:
      mode: fixed
      fields:
        - "id: int"
        - "name: str"
        - "amount: int"
        - "category: str"
    on_validation_failure: discard

# DECIDE: How to route rows
gates:
  - name: amount_threshold
    input: gate_in                  # Receive rows from source
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values           # High amounts → high_values sink
      "false": output               # Normal amounts → output sink

# ACT: Where data goes
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: examples/threshold_gate/output/normal.csv
      schema:
        mode: fixed
        fields:
          - "id: int"
          - "name: str"
          - "amount: int"
          - "category: str"
  high_values:
    plugin: csv
    on_write_failure: discard
    options:
      path: examples/threshold_gate/output/high_values.csv
      schema:
        mode: fixed
        fields:
          - "id: int"
          - "name: str"
          - "amount: int"
          - "category: str"

# Audit trail
landscape:
  url: sqlite:///examples/threshold_gate/runs/audit.db
```

**Key concepts:**

| Section | Purpose |
|---------|---------|
| `source` | **SENSE** - Load data from CSV, validate schema |
| `gates` | **DECIDE** - Route based on condition |
| `sinks` | **ACT** - Write to output files |
| `landscape` | **AUDIT** - Record everything |

**DAG wiring:** Notice `on_success: gate_in` on the source and `input: gate_in` on the gate — these connect pipeline stages by name. Each node declares where it sends rows (`on_success`) and where it receives them (`input`). Sinks require `on_write_failure` to declare what happens if a write fails (`discard` drops the row).

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
elspeth explain --run latest --row 2 --database examples/threshold_gate/runs/audit.db
```

This launches an interactive terminal UI where you can explore:
- The source row and its content hash
- Each processing step (transforms, gates)
- Branch labels, repeated DAG joins, and routing decisions
- Final destination and artifact hash

Use arrow keys to navigate the tree, Enter to update the detail panel, `r` to
refresh, and `q` to quit.

> **Tip:** Use `--no-tui` for plain text output or `--json` for machine-readable output instead of the interactive TUI.

Every decision is traceable. If an auditor asks "why was this transaction flagged?", you have the answer.

---

## Option B: Running in the Web UI

The Web UI is the browser-based authoring and execution surface. It is not a
separate engine: composed pipelines are validated and executed through the same
runtime path as hand-edited YAML, and run evidence is recorded in the configured
audit store.

Use this path when you want to:

- build or revise a pipeline with the Web Composer
- upload source files into a session-scoped file store
- inspect the graph, YAML, validation, audit-readiness, and run outputs in one
  place
- start a background run and watch progress in the browser

> **Note:** The Web Composer requires whatever LLM/provider configuration your
> deployment uses for composition. If you want the fully offline path, use
> [Option A](#option-a-running-locally-python).

### Step 1: Install the Web UI Extra

From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[webui,dev]"
```

### Step 2: Build the Frontend Bundle

```bash
cd src/elspeth/web/frontend
npm install
npm run build
cd ../../../../
```

The FastAPI app serves the built React bundle from
`src/elspeth/web/frontend/dist/`.

### Step 3: Create a Local Demo User

Use a non-default signing key and create a local development account:

```bash
export ELSPETH_WEB__SECRET_KEY="local-dev-secret-key"

python - <<'PY'
import os
from pathlib import Path
from elspeth.web.auth.local import LocalAuthProvider

provider = LocalAuthProvider(
    db_path=Path("data/auth.db"),
    secret_key=os.environ["ELSPETH_WEB__SECRET_KEY"],
)
try:
    provider.create_user(
        user_id="demo",
        password="demo12345",
        display_name="Demo User",
        email="demo@example.com",
    )
    print("Created demo user: demo / demo12345")
except ValueError:
    print("Demo user already exists: demo / demo12345")
PY
```

Do not reuse these credentials outside local development.

### Step 4: Start the Web App

```bash
elspeth web --host 127.0.0.1 --port 8451
```

Open <http://127.0.0.1:8451> and sign in with `demo` / `demo12345`.

### Step 5: Create a Browser-Authored Version

1. Use the session switcher and choose **+ New session**.
2. Keep the default guided mode, or choose **Switch to guided** if your
   account default is freeform.
3. Upload `examples/threshold_gate/input.csv` through the **Files** panel with
   **+ Upload**.
4. Tell the composer:

   ```text
   Use the uploaded transactions CSV as the source. Build a pipeline that routes
   rows with amount > 1000 to a high_values CSV output and all other rows to a
   normal CSV output. Validate it before running.
   ```

5. Review any proposed source, sink, gate, or transform choices before
   accepting them. If the composer asks for field types, keep the same schema
   as the CLI example: `id: int`, `name: str`, `amount: int`, `category: str`.
6. When validation passes, use **Run pipeline**.

The completion bar also exposes **Save for review** and **Export YAML**. Use
**Export YAML** if you want to compare the browser-authored configuration with
`examples/threshold_gate/settings.yaml`.

### Step 6: Inspect the Browser Run

After the run starts, the UI keeps execution in the session:

- progress streams while the run is active
- output artifacts appear in the run results and file surfaces
- diagnostics show validation or runtime failures instead of leaving a silent
  spinner
- audit-readiness and generated YAML stay attached to the composition you ran

The important model is the same as the CLI walkthrough:

```text
source CSV -> threshold decision -> normal/high-value outputs -> audit record
```

---

## Option C: Running with Docker

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
source:
  plugin: csv
  on_success: gate_in
  options:
    path: /app/input/transactions.csv  # Container path!
    schema:
      mode: fixed
      fields:
        - "id: int"
        - "name: str"
        - "amount: int"
        - "category: str"
    on_validation_failure: discard

# DECIDE: Route high-value transactions
gates:
  - name: amount_threshold
    input: gate_in
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": output

# ACT: Write to output files
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: /app/output/normal.csv  # Container path!
      schema:
        mode: fixed
        fields:
          - "id: int"
          - "name: str"
          - "amount: int"
          - "category: str"
  high_values:
    plugin: csv
    on_write_failure: discard
    options:
      path: /app/output/high_values.csv  # Container path!
      schema:
        mode: fixed
        fields:
          - "id: int"
          - "name: str"
          - "amount: int"
          - "category: str"

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

For Docker environments where TUI isn't available, use non-interactive explain output:

```bash
docker run --rm \
  -v $(pwd)/state:/app/state:ro \
  ghcr.io/johnm-dta/elspeth:latest \
  explain --run latest --row 2 --no-tui --database /app/state/audit.db
```

> **Tip:** Use `--no-tui` for plain text output or `--json` for machine-readable output. The TUI requires an interactive terminal, so use these flags in CI/CD environments.

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
docker compose run -it --rm elspeth explain --run latest --row 2 --database /app/state/audit.db
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
- Received rows on its `gate_in` input connection
- Evaluated `row['amount'] > 1000` for each row
- Rows with `true` → routed to `high_values` sink
- Rows with `false` → routed to `output` sink

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

In the Web UI, ask the composer to change the threshold and then review the
updated validation result before using **Run pipeline** again.

### Add a Third Tier

Route "premium" transactions (> $2500) separately:

```yaml
source:
  plugin: csv
  on_success: premium_check_in
  options: ...

gates:
  - name: premium_check
    input: premium_check_in
    condition: "row['amount'] > 2500"
    routes:
      "true": premium
      "false": high_value_check_in    # Chain to next gate

  - name: high_value_check
    input: high_value_check_in
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": output

sinks:
  output:
    # ... normal transactions
  high_values:
    # ... high-value transactions
  premium:
    plugin: csv
    on_write_failure: discard
    options:
      path: /app/output/premium.csv
      schema:
        mode: observed
```

### Add a Transform

Add a field before routing:

```yaml
source:
  plugin: csv
  on_success: mapper_in
  options: ...

transforms:
  - name: add_tier
    plugin: field_mapper
    input: mapper_in
    on_success: tier_gate_in
    on_error: discard
    options:
      schema:
        mode: observed
      computed:
        tier: "row['amount'] > 2500 and 'premium' or (row['amount'] > 1000 and 'high' or 'normal')"

gates:
  - name: tier_router
    input: tier_gate_in
    condition: "row['tier'] == 'premium'"
    routes:
      "true": premium
      "false": output
```

---

## Troubleshooting

For comprehensive troubleshooting, see the [Troubleshooting Guide](troubleshooting.md). Quick fixes for common issues:

### Docker-Specific Issues

- **"File not found"** - See [File Not Found Errors](troubleshooting.md#file-not-found-errors)
- **"Permission denied"** - See [Permission Denied on Output](troubleshooting.md#permission-denied-on-output)

### "Invalid schema" error

**Symptom:** `ValidationError: field 'amount' expected int, got str`

**Fix:** Add type coercion in schema:
```yaml
schema:
  mode: fixed
  fields:
    - "amount: int"  # Will coerce "1500" to 1500
```

### Nothing in high_values.csv

**Symptom:** All rows go to normal.csv

**Fix:** Ensure the source schema coerces numeric fields. The expression parser does NOT allow function calls like `int()`:
```yaml
# In source config - coerce to int at source
source:
  plugin: csv
  on_success: gate_in
  options:
    schema:
      mode: fixed
      fields:
        - "amount: int"  # Coerces "1500" to 1500

# Then in gate condition - amount is already an int
condition: "row['amount'] > 1000"
```

---

## Next Steps

Now that you've built your first pipeline:

1. **Add an LLM transform** - See `examples/openrouter_sentiment/` for LLM classification
2. **Try guided browser authoring** - See [User Manual: Web Composer](user-manual.md#web-composer-guided-mode)
3. **Share a browser-authored pipeline for review** - See [Sharing Pipelines](sharing-pipelines.md)
4. **Export the audit trail** - Add `landscape.export` to create signed exports
5. **Build a custom plugin** - See [PLUGIN.md](../../PLUGIN.md) for plugin development
6. **Explore the architecture** - See [ARCHITECTURE.md](../../ARCHITECTURE.md) for system design

---

## Quick Reference

### Local Commands

```bash
elspeth validate --settings path/to/settings.yaml
elspeth run --settings path/to/settings.yaml --execute
elspeth explain --run latest --row <row_id> --database examples/threshold_gate/runs/audit.db
elspeth plugins list
elspeth plugins list --format json
elspeth plugins inspect source csv
```

### Web UI Commands

```bash
uv pip install -e ".[webui,dev]"

cd src/elspeth/web/frontend
npm install
npm run build
cd ../../../../

export ELSPETH_WEB__SECRET_KEY="local-dev-secret-key"
elspeth web --host 127.0.0.1 --port 8451
```

| UI Control | Description |
|------------|-------------|
| `+ New session` | Start a new browser composition session |
| `+ Upload` | Add a session-scoped source file |
| `Run pipeline` | Execute the validated composition |
| `Export YAML` | Inspect or download generated YAML |
| `Save for review` | Create a shareable read-only review link when validation passes |

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
| `explain --run latest --row N --database <path>` | Explain decision (TUI) |
| `plugins list` | List available plugins |
| `plugins list --format json` | Emit plugin catalog JSON |
| `plugins inspect <type> <name>` | Inspect one plugin schema |
| `--help` | Show all commands |

---

## See Also

- [README.md](../../README.md) - Project overview
- [User Manual](user-manual.md) - CLI and Web Composer reference
- [Sharing Pipelines](sharing-pipelines.md) - Save-for-review and shared inspect flow
- [Docker Guide](docker.md) - Complete Docker deployment
- [PLUGIN.md](../../PLUGIN.md) - Creating custom plugins
- [examples/](../../examples/) - More example pipelines
