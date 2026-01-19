# ELSPETH User Manual

This manual describes how to configure and run ELSPETH pipelines.

> **Related Documentation:**
> - `README.md` - Project overview and quick start
> - `PLUGIN.md` - Plugin development guide
> - `docs/contracts/plugin-protocol.md` - Protocol specification

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Reference](#cli-reference)
3. [Configuration Reference](#configuration-reference)
4. [Pipeline Components](#pipeline-components)
5. [Built-in Plugins](#built-in-plugins)
6. [Examples Walkthrough](#examples-walkthrough)
7. [Audit Trail & Explain](#audit-trail--explain)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/your-org/elspeth-rapid.git
cd elspeth-rapid

# Create virtual environment (uv required)
uv venv
source .venv/bin/activate

# Install for development
uv pip install -e ".[dev]"

# Or with LLM support
uv pip install -e ".[llm]"
```

### Run Your First Pipeline

```bash
# Validate configuration (dry run)
uv run elspeth run -s examples/boolean_routing/settings.yaml --dry-run

# Execute the pipeline
uv run elspeth run -s examples/boolean_routing/settings.yaml --execute

# View results
cat examples/boolean_routing/output/approved.csv
cat examples/boolean_routing/output/rejected.csv
```

### Verifying Pipeline Success

A successful run displays output like:

```
✓ Pipeline completed
  Source: 10 rows loaded
  Sink 'approved': 5 rows written
  Sink 'rejected': 5 rows written
  Audit trail: examples/boolean_routing/runs/audit.db
```

**Success checklist:**

- [ ] Exit code is 0 (no error)
- [ ] Output files exist with expected row counts
- [ ] No `ERROR` or `FAILED` lines in output
- [ ] Audit database created/updated

**Verify with explain:**

```bash
# Check all tokens reached terminal state
uv run elspeth explain -r latest --no-tui
```

All tokens should show a terminal state: `COMPLETED`, `ROUTED`, or `QUARANTINED`.

### Minimal Pipeline Configuration

```yaml
# my_pipeline/settings.yaml

# Data source (exactly one required)
datasource:
  plugin: csv
  options:
    path: input.csv
    schema:
      fields: dynamic
    on_validation_failure: discard

# Output sinks (at least one required)
sinks:
  output:
    plugin: csv
    options:
      path: output/results.csv
      schema:
        fields: dynamic

# Default sink for processed rows
output_sink: output

# Audit trail database
landscape:
  url: sqlite:///runs/audit.db
```

---

## CLI Reference

### `elspeth run`

Execute a pipeline.

```bash
elspeth run -s <settings.yaml> [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--settings` | `-s` | Path to settings YAML file (required) |
| `--execute` | `-x` | Actually run the pipeline (required for safety) |
| `--dry-run` | `-n` | Validate and show what would run without executing |
| `--verbose` | `-v` | Show detailed output |

**Examples:**

```bash
# Dry run - validate configuration
uv run elspeth run -s settings.yaml --dry-run

# Execute pipeline
uv run elspeth run -s settings.yaml --execute

# Execute with verbose output
uv run elspeth run -s settings.yaml --execute --verbose
```

### `elspeth validate`

Validate configuration without running.

```bash
elspeth validate -s <settings.yaml>
```

| Option | Short | Description |
|--------|-------|-------------|
| `--settings` | `-s` | Path to settings YAML file (required) |

### `elspeth explain`

Explain lineage for a row or token (audit trail query).

```bash
elspeth explain -r <run_id> [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--run` | `-r` | Run ID to explain, or `latest` (required) |
| `--row` | | Row ID or index to explain |
| `--token` | `-t` | Token ID for precise lineage |
| `--no-tui` | | Output text instead of interactive TUI |
| `--json` | | Output as JSON |

**Examples:**

```bash
# Interactive TUI for latest run
uv run elspeth explain -r latest

# Text output for specific row
uv run elspeth explain -r latest --row 42 --no-tui

# JSON output for specific token
uv run elspeth explain -r run-abc123 --token tok-xyz789 --json
```

### `elspeth resume`

Resume a failed run from checkpoint.

```bash
elspeth resume <run_id> [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--database` | `-d` | Path to Landscape database file |
| `--settings` | `-s` | Path to settings YAML (default: settings.yaml) |
| `--execute` | `-x` | Actually execute resume (default is dry-run) |

**Examples:**

```bash
# Dry run - show what would resume
uv run elspeth resume run-abc123

# Actually resume processing
uv run elspeth resume run-abc123 --execute

# Resume with explicit database
uv run elspeth resume run-abc123 --database ./landscape.db --execute
```

### `elspeth plugins list`

List available plugins.

```bash
elspeth plugins list
```

Output shows registered sources, transforms, and sinks.

### `elspeth purge`

Purge old payloads to free storage based on retention policies.

```bash
elspeth purge [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--database` | `-d` | Path to Landscape database |
| `--older-than` | | Purge payloads older than N days |
| `--dry-run` | `-n` | Show what would be purged without deleting |
| `--force` | `-f` | Skip confirmation prompt |

**Examples:**

```bash
# Preview what would be purged (payloads older than 30 days)
uv run elspeth purge --database runs/audit.db --older-than 30 --dry-run

# Actually purge old payloads
uv run elspeth purge --database runs/audit.db --older-than 30

# Force purge without confirmation
uv run elspeth purge --database runs/audit.db --older-than 7 --force
```

**Note:** Purging removes payload data but preserves audit metadata (hashes, timestamps). The audit trail remains complete for compliance purposes.

---

## Configuration Reference

Pipeline configuration is defined in a YAML file (typically `settings.yaml`).

### Top-Level Structure

```yaml
# Required: Data source
datasource:
  plugin: <source_plugin_name>
  options: { ... }

# Optional: Row transforms (stateless)
row_plugins:
  - plugin: <transform_plugin_name>
    options: { ... }

# Optional: Gates (routing decisions)
gates:
  - name: <gate_name>
    condition: "<expression>"
    routes:
      "<label>": <destination>

# Optional: Aggregations (batching)
aggregations:
  - name: <aggregation_name>
    plugin: <batch_transform_name>
    trigger: { ... }
    output_mode: single | passthrough | transform
    options: { ... }

# Required: Output sinks
sinks:
  <sink_name>:
    plugin: <sink_plugin_name>
    options: { ... }

# Required: Default output sink
output_sink: <sink_name>

# Required: Audit trail
landscape:
  url: <database_url>
  export: { ... }  # Optional
```

### Datasource Configuration

Every pipeline requires exactly one data source.

```yaml
datasource:
  plugin: csv           # Plugin name (csv, json)
  options:
    path: data/input.csv
    schema:
      mode: strict | free | dynamic
      fields:
        - "field_name: type"
    on_validation_failure: discard | <sink_name>
```

| Field | Required | Description |
|-------|----------|-------------|
| `plugin` | Yes | Source plugin name |
| `options.path` | Yes | Path to data file |
| `options.schema` | Yes | Schema configuration |
| `options.on_validation_failure` | Yes | Where invalid rows go |

### Schema Configuration

Schemas control data validation at plugin boundaries.

```yaml
# Dynamic - accept any fields (no validation)
schema:
  fields: dynamic

# Strict - only declared fields allowed
schema:
  mode: strict
  fields:
    - "id: int"
    - "name: str"
    - "score: float"
    - "active: bool"

# Free - declared fields required, extras allowed
schema:
  mode: free
  fields:
    - "id: int"
    - "value: float"
```

| Mode | Behavior | Extra Fields |
|------|----------|--------------|
| `dynamic` | No validation | Allowed |
| `strict` | Only declared fields | Rejected |
| `free` | Declared required | Allowed |

**Supported Types:**

| Type | Python | Description |
|------|--------|-------------|
| `str` | `str` | Text |
| `int` | `int` | Integer |
| `float` | `float` | Decimal number |
| `bool` | `bool` | True/False |
| `any` | `Any` | No type checking |

### Row Plugins (Transforms)

Stateless transforms that process rows one at a time.

```yaml
row_plugins:
  - plugin: field_mapper
    options:
      schema:
        fields: dynamic
      mappings:
        new_name: old_name
        computed: "row['a'] + row['b']"
      on_error: error_sink  # Optional: where errors go

  - plugin: passthrough
    options:
      schema:
        fields: dynamic
```

### Gates Configuration

Config-driven routing based on conditions.

```yaml
gates:
  - name: quality_check
    condition: "row['score'] >= 0.8"
    routes:
      "true": high_quality_sink   # When condition is true
      "false": continue           # When condition is false → next node

  - name: category_router
    condition: "row['type'] == 'premium'"
    routes:
      "true": premium_sink
      "false": standard_sink
```

**Condition Expression Syntax:**

```python
# Field access
"row['field_name']"
"row.get('optional_field', 'default')"

# Comparisons
"row['score'] > 0.8"
"row['status'] == 'active'"
"row['count'] >= 100"

# Boolean logic
"row['a'] > 0 and row['b'] < 10"
"row['type'] == 'A' or row['type'] == 'B'"
"not row['disabled']"

# Membership
"row['category'] in ['A', 'B', 'C']"
"'keyword' in row['text']"
```

**Route Destinations:**

| Destination | Behavior |
|-------------|----------|
| `continue` | Continue to next node in pipeline |
| `<sink_name>` | Route directly to named sink |

### Aggregations Configuration

Batch processing with triggers.

```yaml
aggregations:
  - name: batch_stats
    plugin: batch_stats
    trigger:
      count: 100              # Fire after N rows
      timeout_seconds: 3600   # Or after N seconds
    output_mode: single       # N inputs → 1 output
    options:
      schema:
        fields: dynamic
      value_field: amount
      group_by: category
```

**Trigger Options:**

| Trigger | Description |
|---------|-------------|
| `count: N` | Fire after N rows accumulated |
| `timeout_seconds: N` | Fire after N seconds elapsed |
| Both | First condition met wins |

**Output Modes:**

| Mode | Input → Output | Use Case |
|------|----------------|----------|
| `single` | N → 1 | Aggregation (sum, count, mean) |
| `passthrough` | N → N | Batch enrichment (same tokens) |
| `transform` | N → M | Deaggregation (new tokens) |

### Sinks Configuration

Output destinations for processed data.

```yaml
sinks:
  results:
    plugin: csv
    options:
      path: output/results.csv
      schema:
        fields: dynamic

  errors:
    plugin: json
    options:
      path: output/errors.json
      schema:
        fields: dynamic

  database_sink:
    plugin: database
    options:
      url: postgresql://user:pass@host/db
      table: results
      schema:
        mode: strict
        fields:
          - "id: int"
          - "result: str"
```

### Landscape (Audit Trail) Configuration

```yaml
landscape:
  url: sqlite:///runs/audit.db

  # Optional: Export audit trail after run
  export:
    enabled: true
    sink: audit_export_sink
    format: json
    sign: false  # Set true for legal/compliance use
```

**Database URLs:**

| Database | URL Format |
|----------|------------|
| SQLite | `sqlite:///path/to/audit.db` |
| PostgreSQL | `postgresql://user:pass@host:port/dbname` |

**Export Options:**

| Option | Description |
|--------|-------------|
| `enabled` | Enable post-run export |
| `sink` | Sink name for export output |
| `format` | `json` (recommended for heterogeneous records) |
| `sign` | HMAC sign records (requires `ELSPETH_SIGNING_KEY` env var) |

---

## Pipeline Components

### Data Flow

```
┌──────────┐    ┌───────────┐    ┌───────┐    ┌──────────┐    ┌──────┐
│  Source  │───▶│ Transform │───▶│ Gate  │───▶│ Aggreg.  │───▶│ Sink │
└──────────┘    └───────────┘    └───────┘    └──────────┘    └──────┘
     │               │               │              │             │
     │               │               │              │             │
  Loads           Processes       Routes        Batches       Outputs
  external         rows           rows          rows          data
  data          (stateless)    to paths       (stateful)
```

### Component Types

| Component | Purpose | Cardinality |
|-----------|---------|-------------|
| **Source** | Load data | Exactly 1 per run |
| **Transform** | Process rows (stateless) | 0 or more |
| **Gate** | Route rows based on conditions | 0 or more |
| **Aggregation** | Batch rows until trigger | 0 or more |
| **Sink** | Output data | 1 or more |

### Token Lifecycle

Every row entering the pipeline becomes a **token** that is tracked through the audit trail:

```
Source Row → Token Created → Transform → Gate → ... → Sink → Terminal State
```

**Terminal States:**

| State | Meaning |
|-------|---------|
| `COMPLETED` | Reached output sink successfully |
| `ROUTED` | Sent to named sink by gate |
| `QUARANTINED` | Failed validation, stored for investigation |
| `CONSUMED_IN_BATCH` | Aggregated into batch output |
| `EXPANDED` | Parent token that was deaggregated |

---

## Built-in Plugins

### Sources

| Plugin | Description | Key Options |
|--------|-------------|-------------|
| `csv` | Load from CSV files | `path`, `delimiter`, `encoding` |
| `json` | Load from JSON/JSONL files | `path`, `format` (json/jsonl) |

### Transforms

| Plugin | Description | Key Options |
|--------|-------------|-------------|
| `passthrough` | Pass rows unchanged | `validate_input` |
| `field_mapper` | Rename/select/compute fields | `mappings`, `include`, `exclude` |
| `json_explode` | Expand array field to rows | `array_field`, `output_field` |
| `batch_stats` | Compute batch statistics | `value_field`, `group_by` |
| `batch_replicate` | Replicate rows (deaggregation) | `copies_field` |

### Sinks

| Plugin | Description | Key Options |
|--------|-------------|-------------|
| `csv` | Write to CSV files | `path`, `delimiter` |
| `json` | Write to JSON/JSONL files | `path`, `format` |
| `database` | Write to database tables | `url`, `table` |

---

## Examples Walkthrough

ELSPETH includes several example pipelines in `examples/`:

### 1. Boolean Routing

Routes rows based on a true/false field.

```bash
uv run elspeth run -s examples/boolean_routing/settings.yaml --execute
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
uv run elspeth run -s examples/threshold_gate/settings.yaml --execute
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
uv run elspeth run -s examples/batch_aggregation/settings.yaml --execute
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
uv run elspeth run -s examples/deaggregation/settings.yaml --execute
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
uv run elspeth run -s examples/json_explode/settings.yaml --execute
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
uv run elspeth run -s examples/audit_export/settings.yaml --execute
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

---

## Audit Trail & Explain

ELSPETH maintains a complete audit trail in the Landscape database.

### What's Recorded

| Record Type | Contents |
|-------------|----------|
| **Run** | Run ID, start/end time, configuration hash, status |
| **Node** | Plugin name, version, determinism, schema hashes |
| **Token** | Row data, state transitions, parent/child linkage |
| **Transform** | Input/output hashes, duration, error details |
| **Artifact** | Content hash, size, location for every sink write |

### Querying Lineage

```bash
# Interactive TUI
uv run elspeth explain -r latest

# Text output for specific row
uv run elspeth explain -r latest --row 42 --no-tui

# JSON output for programmatic access
uv run elspeth explain -r latest --row 42 --json
```

### The Attributability Test

For any output, ELSPETH can prove complete lineage:

```
"Output row X came from source row Y,
 was transformed by nodes [A, B, C],
 with configuration hash D,
 at timestamps [T1, T2, T3]"
```

### Signed Exports (Compliance)

For legal/regulatory compliance, audit exports can be HMAC-signed:

```bash
# Set signing key
export ELSPETH_SIGNING_KEY="your-secret-key"

# Enable in settings.yaml
landscape:
  export:
    enabled: true
    sink: audit_export
    sign: true
```

---

## Troubleshooting

### Common Errors

#### "Schema compatibility error"

```
SchemaCompatibilityError: Database schema version mismatch
```

**Cause:** Old database file with outdated schema.
**Fix:** Delete the `.db` file and re-run to create fresh database.

```bash
rm runs/audit.db
uv run elspeth run -s settings.yaml --execute
```

#### "on_validation_failure is required"

```
PluginConfigError: on_validation_failure must be a sink name or 'discard'
```

**Cause:** Source plugin missing quarantine configuration.
**Fix:** Add `on_validation_failure` to source options:

```yaml
datasource:
  plugin: csv
  options:
    path: input.csv
    schema:
      fields: dynamic
    on_validation_failure: discard  # or a sink name
```

#### "Plugin not found"

```
KeyError: 'my_plugin'
```

**Cause:** Plugin not registered or misspelled.
**Fix:** Check available plugins with `elspeth plugins list`.

#### "Condition evaluation failed"

```
GateConditionError: NameError: name 'row' is not defined
```

**Cause:** Invalid gate condition syntax.
**Fix:** Use correct syntax: `"row['field'] == 'value'"` (note the quotes).

### Debugging Tips

1. **Start with dry-run:**
   ```bash
   uv run elspeth run -s settings.yaml --dry-run
   ```

2. **Use verbose mode:**
   ```bash
   uv run elspeth run -s settings.yaml --execute --verbose
   ```

3. **Check validation:**
   ```bash
   uv run elspeth validate -s settings.yaml
   ```

4. **Examine audit trail:**
   ```bash
   uv run elspeth explain -r latest --no-tui
   ```

5. **Test with simple input:**
   - Start with 5-10 rows
   - Use `schema: {fields: dynamic}` initially
   - Add strictness incrementally

### Performance Considerations

| Scenario | Recommendation |
|----------|----------------|
| Large files | Use streaming sources (not load-all-in-memory) |
| Many transforms | Profile individual transforms |
| Database sinks | Use batch writes, connection pooling |
| Aggregations | Tune batch size to balance memory vs. latency |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ELSPETH_SIGNING_KEY` | Secret key for HMAC-signing audit exports |
| `ELSPETH_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Further Reading

- `CLAUDE.md` - Project architecture and coding standards
- `PLUGIN.md` - How to create new plugins
- `docs/contracts/plugin-protocol.md` - Complete protocol specification
- `TEST_SYSTEM.md` - Testing documentation
