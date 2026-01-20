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
9. [Examples](#examples)

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

### Automatic .env Loading

ELSPETH automatically loads environment variables from a `.env` file when you run any command. This eliminates the need to manually `source .env` before running pipelines.

**How it works:**

1. When any `elspeth` command runs, it looks for `.env` in the current directory
2. If not found, it searches parent directories
3. Variables from `.env` are loaded into the environment
4. Existing environment variables are **not** overwritten

### Example .env File

Create a `.env` file in your project root:

```bash
# .env - ELSPETH environment configuration

# ═══════════════════════════════════════════════════════════════════
# LLM API Keys
# ═══════════════════════════════════════════════════════════════════

# OpenRouter (for openrouter_llm transform)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Azure OpenAI (for azure_llm and azure_batch_llm transforms)
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Azure Content Safety (for azure_content_safety transform)
AZURE_CONTENT_SAFETY_KEY=your-content-safety-key
AZURE_CONTENT_SAFETY_ENDPOINT=https://your-resource.cognitiveservices.azure.com

# ═══════════════════════════════════════════════════════════════════
# ELSPETH Security Settings
# ═══════════════════════════════════════════════════════════════════

# Secret fingerprinting key (REQUIRED for production)
# Used to hash API keys before storing in audit trail
ELSPETH_FINGERPRINT_KEY=your-stable-secret-key

# Signing key for audit exports (optional)
# Enables HMAC signatures on exported audit records
ELSPETH_SIGNING_KEY=your-signing-key

# ═══════════════════════════════════════════════════════════════════
# Development Settings (DO NOT USE IN PRODUCTION)
# ═══════════════════════════════════════════════════════════════════

# Skip secret fingerprinting (development only!)
# ELSPETH_ALLOW_RAW_SECRETS=true
```

### Skipping .env Loading

In CI/CD or containerized environments where secrets are injected externally:

```bash
# Skip .env loading entirely
elspeth --no-dotenv run -s settings.yaml --execute
```

### Security Best Practices

1. **Never commit `.env` to version control**

   Add to `.gitignore`:
   ```gitignore
   .env
   .env.local
   .env.*.local
   ```

2. **Use different keys per environment**

   ```bash
   # Production: Real fingerprint key for audit integrity
   ELSPETH_FINGERPRINT_KEY=prod-key-that-never-changes

   # Development: Can use any value
   ELSPETH_FINGERPRINT_KEY=dev-key
   ```

3. **Set `ELSPETH_FINGERPRINT_KEY` in production**

   Without this, ELSPETH will refuse to run if your config contains API keys. This prevents accidental secret leakage to audit databases.

---

## CLI Commands

### Global Options

```bash
elspeth [OPTIONS] COMMAND [ARGS]

Options:
  --version, -V    Show version and exit
  --no-dotenv      Skip loading .env file
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
elspeth explain --run latest

# Explain a specific run
elspeth explain --run e58480edd52a4292809928bd6425f4ed
```

### Query Specific Rows

```bash
# Explain a specific row
elspeth explain --run latest --row 42

# Explain by token ID (for forked rows)
elspeth explain --run latest --token abc123
```

### Output Formats

```bash
# Interactive TUI (default)
elspeth explain --run latest

# Plain text
elspeth explain --run latest --no-tui

# JSON output
elspeth explain --run latest --json
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

## Examples

### Example 1: Simple CSV Processing

```yaml
# settings.yaml
datasource:
  plugin: csv
  options:
    path: input/data.csv
    schema:
      fields: dynamic

row_plugins:
  - plugin: field_mapper
    options:
      mapping:
        full_name: "{{ row.first_name }} {{ row.last_name }}"

sinks:
  output:
    plugin: csv
    options:
      path: output/processed.csv
      schema:
        fields: dynamic

output_sink: output

landscape:
  url: sqlite:///runs/audit.db
```

```bash
elspeth run -s settings.yaml --execute
```

### Example 2: LLM Sentiment Analysis

```yaml
# settings.yaml
datasource:
  plugin: csv
  options:
    path: input/reviews.csv
    schema:
      fields: dynamic

row_plugins:
  - plugin: openrouter_llm
    options:
      api_key: "${OPENROUTER_API_KEY}"
      model: "openai/gpt-4o-mini"
      template: |
        Analyze sentiment: {{ row.text }}
        Respond with JSON: {"sentiment": "positive/negative/neutral"}
      response_field: analysis

sinks:
  output:
    plugin: csv
    options:
      path: output/analyzed.csv
      schema:
        fields: dynamic

output_sink: output

landscape:
  url: sqlite:///runs/audit.db
```

```bash
# Just run - .env is loaded automatically
elspeth run -s settings.yaml --execute
```

### Example 3: Content Moderation with Routing

```yaml
# settings.yaml
datasource:
  plugin: csv
  options:
    path: input/submissions.csv

gates:
  - name: safety_check
    condition: "row.get('flagged', False)"
    routes:
      "true": review_queue
      "false": continue

sinks:
  approved:
    plugin: csv
    options:
      path: output/approved.csv
  review_queue:
    plugin: csv
    options:
      path: output/needs_review.csv

output_sink: approved

landscape:
  url: sqlite:///runs/audit.db
```

---

## Troubleshooting

### "Secret field found but ELSPETH_FINGERPRINT_KEY is not set"

Your configuration contains API keys but the fingerprint key isn't set:

```bash
# Option 1: Set fingerprint key (production)
export ELSPETH_FINGERPRINT_KEY="your-key"

# Option 2: Allow raw secrets (development only)
export ELSPETH_ALLOW_RAW_SECRETS=true
```

Or add to `.env`:
```bash
ELSPETH_FINGERPRINT_KEY=your-key
```

### "Unknown plugin: xyz"

Check available plugins:
```bash
elspeth plugins list
```

Plugin names are case-sensitive and must match exactly.

### API Authentication Errors (401/403)

1. Check your `.env` file has the correct API key
2. Verify the key is valid and not expired
3. Ensure `.env` is in the current directory or a parent

```bash
# Debug: Check if .env is being loaded
echo $OPENROUTER_API_KEY  # Should be empty before running
elspeth run -s settings.yaml --execute  # .env loads here
```

### Pipeline Hangs or Times Out

Check rate limiting and timeout configuration:
```yaml
concurrency:
  max_workers: 4
  rate_limit:
    calls_per_minute: 60
```

For LLM transforms, increase timeout:
```yaml
row_plugins:
  - plugin: azure_llm
    options:
      timeout: 120  # seconds
```

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
