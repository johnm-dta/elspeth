# ELSPETH User Manual

This manual covers day-to-day usage of the ELSPETH CLI for running auditable pipelines.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Environment Configuration](#environment-configuration)
3. [CLI Commands](#cli-commands)
4. [Running Pipelines](#running-pipelines)
5. [Viewing Available Plugins](#viewing-available-plugins)
6. [Explaining Pipeline Results](#explaining-pipeline-results)
7. [Managing Storage](#managing-storage)
8. [Resuming Failed Runs](#resuming-failed-runs)
9. [Health Checks](#health-checks)
10. [Examples](#examples)

---

## Getting Started

### Installation

```bash
# Clone and install
git clone https://github.com/tachyon-beep/elspeth-rapid.git
cd elspeth-rapid
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"  # Full installation with LLM support
```

### Verify Installation

```bash
elspeth --version
elspeth --help
```

---

## Environment Configuration

See [Environment Variables Reference](reference/environment-variables.md) for the complete list of supported variables, including LLM provider keys, Azure service credentials, and security settings.

**Quick start:** Copy `.env.example` to `.env` (or create a new `.env` file) and fill in your API keys. ELSPETH automatically loads `.env` files from the current or parent directories.

---

## CLI Commands

### Global Options

```bash
elspeth [OPTIONS] COMMAND [ARGS]

Options:
  --version, -V    Show version and exit
  --no-dotenv      Skip loading .env file
  --env-file PATH  Path to .env file (skips automatic search)
  --verbose, -v    Enable verbose/debug logging
  --json-logs      Output structured JSON logs (for machine processing)
  --help           Show help message
```

### Available Commands

| Command | Description |
|---------|-------------|
| `run` | Execute a pipeline |
| `validate` | Validate configuration without running |
| `explain` | Explain lineage for a row or token |
| `plugins list` | List available plugins |
| `purge` | Delete old payloads to free storage |
| `resume` | Resume a failed run from checkpoint |
| `health` | Check system health for deployment verification |

---

## Running Pipelines

### Validate First

Always validate your configuration before running:

```bash
elspeth validate --settings settings.yaml
```

Output shows:
- Source plugin and configuration
- Number of transforms
- Configured sinks
- Graph structure (nodes and edges)

### Execute a Pipeline

```bash
# Dry run - show what would happen
elspeth run --settings settings.yaml --dry-run

# Actually execute (requires explicit --execute flag)
elspeth run --settings settings.yaml --execute

# With verbose output
elspeth run --settings settings.yaml --execute --verbose
```

### Run Output

```
Run completed: RunStatus.COMPLETED
  Rows processed: 100
  Run ID: e58480edd52a4292809928bd6425f4ed
```

The **Run ID** is your key for querying the audit trail later.

---

## Viewing Available Plugins

### List All Plugins

```bash
elspeth plugins list
```

Output:
```
SOURCES:
  csv                  - Load rows from a CSV file.
  json                 - Load rows from a JSON file.
  null                 - A source that yields no rows.
  azure_blob           - Load rows from Azure Blob Storage.

TRANSFORMS:
  passthrough          - Pass rows through unchanged.
  field_mapper         - Map, rename, and select row fields.
  json_explode         - Explode a JSON array field into multiple rows.
  keyword_filter       - Filter rows containing blocked content patterns.
  azure_content_safety - Analyze content using Azure Content Safety API.
  azure_prompt_shield  - Detect jailbreak attempts and prompt injection.
  azure_llm            - LLM transform using Azure OpenAI.
  azure_batch_llm      - Batch LLM transform using Azure OpenAI Batch API.
  openrouter_llm       - LLM transform using OpenRouter API.
  batch_stats          - Compute aggregate statistics over a batch.
  batch_replicate      - Replicate rows based on a copies field.

SINKS:
  csv                  - Write rows to a CSV file.
  json                 - Write rows to a JSON file.
  database             - Write rows to a database table.
  azure_blob           - Write rows to Azure Blob Storage.
```

### Filter by Type

```bash
elspeth plugins list --type source
elspeth plugins list --type transform
elspeth plugins list --type sink
```

---

## Explaining Pipeline Results

### Query by Run ID

```bash
# Explain the latest run
elspeth explain --run latest --database <path/to/audit.db>

# Explain a specific run
elspeth explain --run e58480edd52a4292809928bd6425f4ed --database <path/to/audit.db>
```

### Query Specific Rows

```bash
# Explain a specific row
elspeth explain --run latest --row 42 --database <path/to/audit.db>

# Explain by token ID (for forked rows)
elspeth explain --run latest --token abc123 --database <path/to/audit.db>
```

### Output Formats

```bash
# Interactive TUI (default)
elspeth explain --run latest --database <path/to/audit.db>

# Plain text
elspeth explain --run latest --no-tui --database <path/to/audit.db>

# JSON output
elspeth explain --run latest --json --database <path/to/audit.db>
```

---

## Managing Storage

### Purge Old Payloads

Over time, payload storage grows. Purge old data while preserving audit metadata:

```bash
# See what would be deleted (dry run)
elspeth purge --dry-run --retention-days 90

# Actually delete (with confirmation)
elspeth purge --retention-days 90

# Skip confirmation prompt
elspeth purge --retention-days 90 --yes

# Specify database path explicitly
elspeth purge --database ./runs/audit.db --retention-days 30
```

**Note:** Purging deletes payload blobs but preserves hashes in the audit trail. You can still verify what data existed, you just can't retrieve the content.

---

## Resuming Failed Runs

If a run fails (e.g., API timeout, network error), you can resume from the last checkpoint:

### Check Resume Status

```bash
# Dry run - show resume information
elspeth resume run-abc123

Output:
  Run run-abc123 can be resumed.
  Resume point:
    Token ID: token-xyz
    Node ID: transform_2
    Sequence number: 45
    Unprocessed rows: 55
```

### Execute Resume

```bash
elspeth resume run-abc123 --execute
```

Resume mode:
- Uses `NullSource` (data comes from stored payloads)
- Appends to existing output files (doesn't overwrite)
- Continues from last successful checkpoint

---

## Health Checks

The `health` command verifies system readiness for deployment:

```bash
# Basic health check
elspeth health

# Verbose output with details
elspeth health --verbose

# JSON output (for automation)
elspeth health --json
```

### Health Check Options

| Option | Description |
|--------|-------------|
| `--verbose, -v` | Include detailed check information |
| `--json, -j` | Output as JSON |

### What Gets Checked

- **version**: ELSPETH version
- **commit**: Git commit SHA (if available)
- **python**: Python version
- **database**: Database connectivity (if `DATABASE_URL` is set)
- **plugins**: Plugin availability

### Example JSON Output

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "commit": "abc123f",
  "checks": {
    "version": {"status": "ok", "value": "0.1.0"},
    "python": {"status": "ok", "value": "3.11.9"},
    "database": {"status": "ok", "value": "connected"},
    "plugins": {"status": "ok", "value": "4 sources, 11 transforms, 4 sinks"}
  }
}
```

---

## Examples Walkthrough

ELSPETH includes several example pipelines in `examples/`:

### 1. Boolean Routing

Routes rows based on a true/false field.

```bash
elspeth run -s examples/boolean_routing/settings.yaml --execute
```

**Input:** CSV with `approved` column (true/false)
**Output:** Separate CSVs for approved and rejected rows

**Verify:**
```bash
wc -l examples/boolean_routing/output/*.csv
#   6 approved.csv   (5 data rows + header)
#   6 rejected.csv   (5 data rows + header)
```

### 2. Threshold Gate

Routes high-value transactions to separate output.

```bash
elspeth run -s examples/threshold_gate/settings.yaml --execute
```

**Input:** CSV with `amount` column
**Output:** High values (>1000) and normal values in separate files

**Verify:**
```bash
cat examples/threshold_gate/output/high_values.csv | head -3
# id,amount,description
# 2,1500,Large purchase
# 4,2000,Premium service
```

### 3. Batch Aggregation

Computes statistics over batches of rows.

```bash
elspeth run -s examples/batch_aggregation/settings.yaml --execute
```

**Input:** 15 transactions
**Output:** 3 batch summaries (one per 5 rows, grouped by category)

**Verify:**
```bash
cat examples/batch_aggregation/output/batch_summaries.csv
# category,count,sum,mean
# electronics,5,2750,550.0
# clothing,5,1250,250.0
# groceries,5,375,75.0
```

### 4. Deaggregation

Demonstrates N→M row expansion with new tokens.

```bash
elspeth run -s examples/deaggregation/settings.yaml --execute
```

**Input:** 6 rows with `copies` field (values: 2,1,3,2,1,2 = 11 total)
**Output:** 11 rows (each replicated by its copies value)

**Verify:**
```bash
wc -l examples/deaggregation/output/replicated.csv
# 12 (11 data rows + header)

head -4 examples/deaggregation/output/replicated.csv
# id,name,copies,category,copy_index
# 1,Alice,2,standard,0
# 1,Alice,2,standard,1
# 2,Bob,1,premium,0
```

### 5. JSON Explode

Expands array fields into individual rows.

```bash
elspeth run -s examples/json_explode/settings.yaml --execute
```

**Input:** 3 orders with `items` arrays
**Output:** 6 rows (one per item)

**Verify:**
```bash
cat examples/json_explode/output/order_items.json | head -20
# Shows individual items with order_id, item details, and item_index
```

### 6. Audit Export

Exports complete audit trail to JSON for compliance.

```bash
elspeth run -s examples/audit_export/settings.yaml --execute
```

**Input:** 8 submissions
**Output:** Routed results + complete audit trail JSON

**Verify:**
```bash
# Check routed outputs
wc -l examples/audit_export/output/*.csv
#   5 corporate.csv      (4 data rows + header)
#   5 non_corporate.csv  (4 data rows + header)

# Check audit trail exists and has content
ls -la examples/audit_export/output/audit_trail.json
# Should show non-zero file size
```

### Additional Examples

For more complex scenarios, see the configuration reference:

- **LLM Sentiment Analysis** - Using `openrouter_llm` plugin with templates
- **Content Moderation with Routing** - Gates with condition expressions
- **Fork/Join Patterns** - Parallel processing with coalesce

See [Configuration Reference](reference/configuration.md) for the complete settings documentation.

---

## Troubleshooting

For comprehensive troubleshooting, see the [Troubleshooting Guide](guides/troubleshooting.md).

### Quick Fixes

**"ELSPETH_FINGERPRINT_KEY is not set"** - Set the key or allow raw secrets for development:
```bash
export ELSPETH_FINGERPRINT_KEY="your-key"
# OR for development only:
export ELSPETH_ALLOW_RAW_SECRETS=true
```

**"Unknown plugin: xyz"** - Check available plugins with `elspeth plugins list` (names are case-sensitive).

**Pipeline hangs** - Run with `--verbose` to identify the bottleneck and check your rate limit configuration.

---

## Getting Help

```bash
# General help
elspeth --help

# Command-specific help
elspeth run --help
elspeth plugins --help
elspeth explain --help
```

For bug reports and feature requests, see the project repository.
