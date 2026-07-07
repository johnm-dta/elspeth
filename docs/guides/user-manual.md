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
10. [Examples](#examples-walkthrough)
11. [Web Composer: Guided Mode](#web-composer-guided-mode)

---

## Getting Started

### Installation

```bash
# Clone and install
git clone https://github.com/johnm-dta/elspeth.git
cd elspeth
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

See [Environment Variables Reference](../reference/environment-variables.md) for the complete list of supported variables, including LLM provider keys, Azure service credentials, and security settings.

**Quick start:** Create a `.env` file if you want local environment-based configuration, then fill in the required keys. ELSPETH automatically loads `.env` files from the current or parent directories.

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
  --install-completion  Install completion for the current shell
  --show-completion     Show completion script for the current shell
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
| `web` | Start the web application server |

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

# JSON output (for machine processing)
elspeth run --settings settings.yaml --execute --format json
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
  azure_blob           - Load rows from Azure Blob Storage.
  csv                  - Load rows from a CSV file.
  dataverse            - Load rows from Microsoft Dataverse via OData v4 REST API.
  json                 - Load rows from a JSON file.
  null                 - A source that yields no rows.
  text                 - Load one output row per text line into a configured column.

TRANSFORMS:
  azure_document_intelligence - Enrich rows with Azure AI Document Intelligence extraction.
  blob_csv_expand     - Expand a payload-store CSV blob into rows.
  blob_fetch          - Fetch an operator-authorised remote document into the payload store.
  batch_replicate      - Replicate rows based on a copies field.
  batch_stats          - Compute aggregate statistics over a batch, optionally per group_by value.
  field_mapper         - Map, rename, and select row fields.
  json_explode         - Explode a JSON array field into multiple rows.
  keyword_filter       - Filter rows containing blocked content patterns.
  passthrough          - Pass rows through unchanged.
  report_assemble      - Assemble a batch of text rows into one report row with pagination metadata.
  truncate             - Truncate string fields to specified maximum lengths.
  type_coerce          - Perform explicit, strict, per-field type normalization.
  value_transform      - Apply expressions to compute new or modified field values.
  web_scrape           - Fetch webpages, extract content, generate fingerprints.
  azure_content_safety - Analyze content using Azure Content Safety API.
  azure_prompt_shield  - Detect jailbreak attempts and prompt injection using Azure Prompt Shield.
  llm                  - Unified LLM transform with provider dispatch and strategy selection.
  rag_retrieval        - Enriches rows with retrieval-augmented context from search providers.

SINKS:
  azure_blob           - Write rows to Azure Blob Storage.
  chroma_sink          - Write rows to a Chroma vector database.
  csv                  - Write rows to a CSV file.
  database             - Write rows to a database table.
  dataverse            - Write rows to Microsoft Dataverse via OData v4 REST API.
  json                 - Write rows to a JSON file.
```

### Filter by Type

```bash
elspeth plugins list --type source
elspeth plugins list --type transform
elspeth plugins list --type sink
```

### Machine-Readable Catalog

```bash
elspeth plugins list --format json
elspeth plugins list --type source --format json
elspeth plugins inspect source csv
elspeth plugins inspect source csv --format json
```

`plugins inspect` shows the catalog description, config fields, JSON Schema,
and composer knob schema for one plugin.

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

# Plain text (for non-interactive terminals or CI/CD)
elspeth explain --run latest --no-tui --database <path/to/audit.db>

# JSON output
elspeth explain --run latest --json --database <path/to/audit.db>

# Disambiguate when a row has multiple terminal tokens (e.g., forked rows)
elspeth explain --run latest --row 42 --sink high_values --database <path/to/audit.db>
```

The interactive TUI shows a selectable lineage tree and detail panel. Use arrow
keys to move through run, branch, node, token, and status rows; press Enter to
update the detail panel; press `r` to refresh and `q` to quit. Use `--row`,
`--token`, and `--sink` to focus the initial lineage view. In non-interactive
terminals or CI, prefer `--no-tui` or `--json`.

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

# Specify database and payload directory explicitly
elspeth purge --database ./runs/audit.db --payload-dir ./runs/payloads --retention-days 30
```

**Note:** Purging deletes payload blobs but preserves hashes in the audit trail. You can still verify what data existed, you just can't retrieve the content.

---

## Resuming Failed Runs

If a run fails (e.g., API timeout, network error), you can resume from the last checkpoint:

### Check Resume Status

```bash
# Dry run - show resume information (positional run_id argument)
elspeth resume run-abc123 --settings settings.yaml --database ./runs/audit.db

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
elspeth resume run-abc123 --execute --settings settings.yaml --database ./runs/audit.db

# JSON output
elspeth resume run-abc123 --execute --format json
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
  "version": "0.7.0",
  "commit": "abc123f",
  "checks": {
    "version": {"status": "ok", "value": "0.7.0"},
    "python": {"status": "ok", "value": "3.13.1"},
    "database": {"status": "ok", "value": "connected"},
    "plugins": {"status": "ok", "value": "6 sources, 19 transforms, 6 sinks"}
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

- **LLM Sentiment Analysis** - Using `llm` plugin (with provider: openrouter) and templates
- **Content Moderation with Routing** - Gates with condition expressions
- **Fork/Join Patterns** - Parallel processing with coalesce

See [Configuration Reference](../reference/configuration.md) for the complete settings documentation.

---

## Web Composer: Guided Mode

The Web Composer is a browser-based authoring surface for building ELSPETH
pipelines without hand-editing YAML. Start it with:

```bash
elspeth web
```

Then open the URL printed on the console (typically <http://localhost:8765>).

When you create a new session, the composer starts in **Guided Mode** unless
your Composer preference says otherwise. In 0.7.0 guided mode is LLM-primary:
the browser sends the operator's stage instruction to `/guided/chat`, and the
server applies the model's proposed source, sink, transform, or wiring change to
the in-progress pipeline only after validation.

Guided sessions are created through `POST /guided/start`. The response carries
a closed-enum `WorkflowProfile` so ELSPETH can distinguish a normal guided
session from the passive first-run tutorial. Tutorial profile state is stripped
on fork so it cannot leak into an ordinary session.

### What guided mode is for

Guided mode builds a linear pipeline in ordered stages:

1. **Source** — describe where the data comes from. The source driver can revise
   a committed source in place and can route a URL-row source to the web-scrape
   recipe when that shape is appropriate.
2. **Sink** — describe where results should land and which output fields matter.
   The sink driver supports free-text intent and commits the resulting sink
   configuration only after validation.
3. **Transforms** — describe how to bridge the source to the sink. The transform
   stage may apply a recipe-backed path or a model-proposed transform chain, but
   the committed pipeline still passes the same runtime-oriented validators as
   YAML.
4. **Wiring** — review the final graph shape. `STEP_4_WIRE` rebuilds edges from
   model connection labels, renders the contract overlay, and accepts only a
   valid `CONFIRM_WIRING` payload.

Each stage can be revised against its current state. Revision context is passed
back to the model so it amends the committed stage rather than starting from a
blank proposal.

### When to use guided mode versus freeform

- **Use guided mode** if you want a structured conversation that builds and
  verifies the pipeline one stage at a time.
- **Use freeform mode** if you already know which plugins you want to wire
  together, or if your pipeline does not match any of the patterns guided mode
  supports (multi-source pipelines, custom branching topologies, exotic
  aggregations, etc.).

Both modes target the same runtime, the same validators, and the same audit
trail. Switching modes never discards pipeline state — only the authoring
surface changes.

### Validation, interpretation, and sign-off

The LLM proposes changes, but it is not the authority. ELSPETH validates and
persists the resulting pipeline state, then shows a plain-language gloss,
validation summary, and graph impact for review.

If a stage depends on a subjective interpretation, guided mode surfaces a
pending interpretation card and blocks advancement until the card is reviewed.
At the final wiring stage, the advisor sign-off path can return
`REQUEST_ADVISOR`; that re-emits the wire turn for review rather than
auto-completing the pipeline.

### Completion and execution

When the wizard runs out of guided steps, you see a **completion summary**
showing the final pipeline shape. From the summary you can:

- **Save and exit** — keep the pipeline as it is; the session is saved.
- **Drop to freeform to keep editing** — switch to freeform mode with the
  completed pipeline pre-loaded, and continue authoring there.

Once the pipeline is saved you can validate it, preview the YAML, and execute
it directly from the composer. The composer's `/validate` and `/execute`
endpoints use the same runtime assembly and graph validation contracts as
`elspeth validate` and `elspeth run` — there is no separate UI-only validator.

### What guided mode does not cover

Guided mode is intentionally narrow. It does not (yet) cover:

- Pipelines with multiple sources.
- Pipelines with branching topologies (forks, gates with multiple downstream
  paths, fork+coalesce patterns).
- Complex aggregation workflows that require custom topology.
- Custom plugin authoring.

For any of these, exit to freeform when guided mode reaches a step it cannot
represent. The chat history and the partial pipeline state both carry over.

### See also

- Historical guided-mode technical design material is preserved in git history
  or maintainer-local archives.
- For freeform composer authoring, plugin discovery, and tool contracts, see
  the composer skill at `src/elspeth/web/composer/skills/pipeline_composer.md`.

---

## Troubleshooting

For comprehensive troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).

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

For bug reports and feature requests, see the project repository. Include a
sanitized reproduction, not raw session data. Do not attach raw composer chat
history or session exports; remove secrets, tokens, PII, blob contents, sample
rows, URLs, and organization-specific identifiers before posting.
