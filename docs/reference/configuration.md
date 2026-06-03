# Configuration Reference

Complete reference for ELSPETH pipeline configuration.

---

## Table of Contents

- [Configuration File Format](#configuration-file-format)
- [Top-Level Settings](#top-level-settings)
- [Secrets Settings](#secrets-settings)
- [Source Settings](#source-settings)
- [Sink Settings](#sink-settings)
- [Transform Settings](#transform-settings)
- [Gate Settings](#gate-settings)
- [Aggregation Settings](#aggregation-settings)
- [Coalesce Settings](#coalesce-settings)
- [Pipeline Dependencies](#pipeline-dependencies)
- [Commencement Gates](#commencement-gates)
- [Collection Probes](#collection-probes)
- [Landscape Settings (Audit Trail)](#landscape-settings-audit-trail)
- [Concurrency Settings](#concurrency-settings)
- [Rate Limit Settings](#rate-limit-settings)
- [Telemetry Settings](#telemetry-settings)
- [Checkpoint Settings](#checkpoint-settings)
- [Retry Settings](#retry-settings)
- [Payload Store Settings](#payload-store-settings)
- [Environment Variables](#environment-variables)
- [Expression Syntax](#expression-syntax)
- [Complete Example](#complete-example)

---

## Configuration File Format

ELSPETH uses YAML configuration files with environment variable expansion:

```yaml
# Standard variable expansion
database_url: ${DATABASE_URL}

# With default value
database_url: ${DATABASE_URL:-sqlite:///./audit.db}
```

Configuration is loaded with this precedence (highest first):
1. Environment variables (`ELSPETH_*`)
2. Config file (settings.yaml)
3. Pydantic schema defaults

Nested environment variables use double underscore: `ELSPETH_LANDSCAPE__URL`.

---

## Top-Level Settings

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source` | object | **Yes** | - | Source plugin configuration (exactly one per run) |
| `sinks` | object | **Yes** | - | Named sink configurations (at least one required) |
| `run_mode` | string | No | `"live"` | Execution mode: `live`, `replay`, `verify` |
| `replay_from` | string | No | - | Run ID to replay/verify against (required for replay/verify modes) |
| `transforms` | list | No | `[]` | Ordered transforms to apply |
| `gates` | list | No | `[]` | Config-driven routing gates |
| `coalesce` | list | No | `[]` | Fork path merge configurations |
| `aggregations` | list | No | `[]` | Batch processing configurations |
| `depends_on` | list | No | `[]` | Pipeline dependencies — run these before the main pipeline |
| `commencement_gates` | list | No | `[]` | Go/no-go conditions evaluated after dependencies complete |
| `collection_probes` | list | No | `[]` | Vector store collections to probe for gate context |
| `landscape` | object | No | (defaults) | Audit trail configuration |
| `concurrency` | object | No | (defaults) | Parallel processing settings |
| `retry` | object | No | (defaults) | Retry behavior settings |
| `payload_store` | object | No | (defaults) | Large blob storage settings |
| `checkpoint` | object | No | (defaults) | Crash recovery settings |
| `rate_limit` | object | No | (defaults) | External call rate limiting |
| `telemetry` | object | No | (defaults) | Telemetry export configuration |
| `secrets` | object | No | (defaults) | Secret loading configuration |

### Run Modes

| Mode | Behavior |
|------|----------|
| `live` | Execute normally, make real external calls |
| `replay` | Use recorded responses from a previous run |
| `verify` | Compare new results against a previous run |

---

## Secrets Settings

Configure how secrets (API keys, tokens) are loaded for the pipeline.

```yaml
secrets:
  source: keyvault
  vault_url: https://my-vault.vault.azure.net  # Must be literal URL, no ${VAR}
  mapping:
    AZURE_OPENAI_KEY: azure-openai-key
    AZURE_OPENAI_ENDPOINT: openai-endpoint
    ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source` | string | No | `"env"` | Secret source: `env` or `keyvault` |
| `vault_url` | string | When `source: keyvault` | - | Azure Key Vault URL (must be literal HTTPS URL) |
| `mapping` | object | When `source: keyvault` | - | Env var name → Key Vault secret name |

### Source Options

| Source | Behavior |
|--------|----------|
| `env` | Secrets come from environment variables / .env file (default) |
| `keyvault` | Secrets loaded from Azure Key Vault using explicit mapping |

**Important:** `vault_url` must be a **literal URL** like `https://my-vault.vault.azure.net`. Environment variable references like `${AZURE_KEYVAULT_URL}` are **not supported** because secrets must be loaded before environment variable resolution occurs.

### Authentication (Key Vault)

Uses Azure DefaultAzureCredential, which tries (in order):
1. Managed Identity (Azure VMs, App Service, AKS)
2. Azure CLI (`az login`)
3. Environment variables (`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`)
4. Visual Studio Code Azure extension

### Examples

**Local Development:**

```yaml
# Use .env file for local development
secrets:
  source: env
```

**Production with Key Vault:**

```yaml
secrets:
  source: keyvault
  vault_url: https://prod-vault.vault.azure.net
  mapping:
    AZURE_OPENAI_KEY: azure-openai-api-key
    AZURE_OPENAI_ENDPOINT: azure-openai-endpoint
    ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key
```

---

## Source Settings

Configures the single data source for the pipeline.

```yaml
source:
  plugin: csv
  on_success: source_out    # Explicit output connection name
  options:
    path: data/input.csv
    schema:
      mode: fixed
      fields:
        - "id: int"
        - "amount: int"
    on_validation_failure: quarantine
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plugin` | string | **Yes** | Plugin name: `csv`, `json`, `text`, `azure_blob`, `dataverse`, `null` |
| `on_success` | string | **Yes** | Connection name for source output (transforms reference this via `input`) |
| `options` | object | No | Plugin-specific configuration |

### Available Source Plugins

| Plugin | Purpose |
|--------|---------|
| `csv` | Load from CSV file |
| `json` | Load from JSON file or JSONL |
| `text` | Load one output row per text line into a configured column |
| `azure_blob` | Load from Azure Blob Storage |
| `dataverse` | Load from Microsoft Dataverse via OData v4 REST API |
| `null` | Empty source (for testing) |

### Schema Options

```yaml
schema:
  mode: fixed          # fixed, flexible, or observed
  fields:
    - "id: int"       # Field name and type
    - "name: str"
    - "amount: float"
on_validation_failure: quarantine  # quarantine or discard
```

| Schema Mode | Behavior |
|-------------|----------|
| `fixed` | Require exactly the specified fields (extras rejected) |
| `flexible` | At least these fields must be present (extras allowed) |
| `observed` | Infer schema from data (no explicit field definitions) |

### Schema Contracts (DAG Validation)

For observed schemas that still have field requirements, use contract fields:

```yaml
# Producer guarantees these fields will exist in output
schema:
  mode: observed
  guaranteed_fields: [customer_id, timestamp, amount]

# Consumer requires these fields in input
schema:
  mode: observed
  required_fields: [customer_id, amount]
```

| Field | Purpose |
|-------|---------|
| `guaranteed_fields` | Fields the producer guarantees will exist (for observed schemas) |
| `required_fields` | Fields the consumer requires in input (for observed schemas) |

The DAG validates at construction time that upstream `guaranteed_fields` satisfy downstream `required_fields`. For explicit schemas (`mode: fixed` or `flexible`), declared fields are implicitly guaranteed.

---

## Sink Settings

Named output destinations. At least one required.

```yaml
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output/results.csv
      collision_policy: fail_if_exists
      schema:
        mode: observed

  flagged:
    plugin: csv
    on_write_failure: quarantine
    options:
      path: output/flagged.csv
      collision_policy: fail_if_exists
      schema:
        mode: observed

  quarantine:
    plugin: csv
    on_write_failure: discard
    options:
      path: output/quarantine.csv
      collision_policy: fail_if_exists
      schema:
        mode: observed
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plugin` | string | **Yes** | Plugin name: `csv`, `json`, `database`, `azure_blob`, `dataverse`, `chroma_sink` |
| `on_write_failure` | string | **Yes** | Per-row write failure handling: `discard` to drop with audit record, or a sink name to divert to a failsink |
| `options` | object | No | Plugin-specific configuration |

For local file sinks (`csv`, `json`), `options.collision_policy` can make output-path collisions explicit:

| Policy | Use with | Behavior |
|--------|----------|----------|
| `fail_if_exists` | `mode: write` | Refuse to write if the requested output path already exists |
| `auto_increment` | `mode: write` | Pick a free sibling path such as `results-1.json` |
| `append_or_create` | `mode: append` | Append to an existing output or create it if missing |

### Available Sink Plugins

| Plugin | Purpose |
|--------|---------|
| `csv` | Write to CSV file |
| `json` | Write to JSON file |
| `database` | Write to SQL database |
| `azure_blob` | Write to Azure Blob Storage |
| `dataverse` | Write to Microsoft Dataverse via OData v4 REST API |
| `chroma_sink` | Write to a ChromaDB vector database |

---

## Transform Settings

Ordered list of transforms applied to each row. Each transform declares its position in the DAG via `input` (where data comes from) and `on_success` (where successful rows go).

```yaml
transforms:
  - name: enricher
    plugin: field_mapper
    input: source_out
    on_success: output
    on_error: quarantine
    options:
      schema:
        mode: observed
      mappings:
        old_field: new_field
      computed:
        full_name: "row['first_name'] + ' ' + row['last_name']"
      required_input_fields: [first_name, last_name]  # Validated at DAG construction
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique node identifier (human-readable, used in audit trail) |
| `plugin` | string | **Yes** | Plugin name |
| `input` | string | **Yes** | Connection name to receive data from (source `on_success` or another transform's `on_success`) |
| `on_success` | string | **Yes** | Where successful rows go (sink name or connection name for downstream node) |
| `on_error` | string | **Yes** | Sink name for rows that fail processing, or `discard` |
| `options` | object | No | Plugin-specific configuration |
| `options.required_input_fields` | list | No | Fields this transform requires in input (enables DAG validation) |

### Required Input Fields

Transforms can declare which fields they require, enabling the DAG to catch missing field errors at configuration time:

```yaml
transforms:
  - plugin: llm_classifier
    options:
      required_input_fields: [customer_id, message_text]
      # ... other options
```

For template-based transforms (like LLM transforms), use `elspeth.core.templates.extract_jinja2_fields()` to discover which fields your template references:

```python
from elspeth.core.templates import extract_jinja2_fields

template = "Customer {{ row.customer_id }}: {{ row.message_text }}"
fields = extract_jinja2_fields(template)  # frozenset({'customer_id', 'message_text'})
# Add these to required_input_fields in your config
```

### Available Transform Plugins

| Plugin | Purpose |
|--------|---------|
| `passthrough` | Pass rows unchanged |
| `field_mapper` | Rename, compute, drop fields |
| `truncate` | Limit string field lengths |
| `keyword_filter` | Filter rows by regex patterns |
| `json_explode` | Expand list-valued fields to multiple rows |
| `batch_stats` | Compute statistics over a batch, optionally one row per `group_by` value |
| `batch_replicate` | Batch deaggregation; configure as an aggregation with `output_mode: transform` |
| `batch_distribution_profile` | Numeric-only batch distribution statistics; use `batch_top_k` for categorical counts/frequencies |
| `report_assemble` | Assemble a batch of text rows into one report/text row with pagination metadata |
| `web_scrape` | HTML content extraction with SSRF prevention |
| `llm` | Unified LLM transform (azure/openrouter providers, single/multi-query) |
| `azure_content_safety` | Detect harmful content via Azure AI |
| `azure_prompt_shield` | Detect prompt injection via Azure AI |
| `rag_retrieval` | Enriches rows with retrieval-augmented context from search providers |

---

## Gate Settings

Config-driven routing based on expressions. Gates evaluate conditions and route rows to sinks or forward them.

```yaml
gates:
  - name: quality_check
    input: enriched
    condition: "row['confidence'] >= 0.85"
    routes:
      "true": next_step_in   # Connection name for downstream node
      "false": discard       # Virtual terminal discard; records gate_discarded

  - name: amount_threshold
    input: validated
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": output
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique gate identifier |
| `input` | string | **Yes** | Connection name to receive data from |
| `condition` | string | **Yes** | Expression to evaluate (see [Expression Syntax](#expression-syntax)) |
| `routes` | object | **Yes** | Maps evaluation results to destinations |
| `fork_to` | list | No | Branch paths for fork operations |

### Route Destinations

| Destination | Behavior |
|-------------|----------|
| `<connection_name>` | Forward to a downstream node that declares this as its `input` |
| `<sink_name>` | Route directly to named sink |
| `fork` | Split to multiple paths (requires `fork_to`) |
| `discard` | Stop this branch without a sink; records an audited `gate_discarded` terminal outcome |

All route destinations must be explicit connection names, sink names, `fork`, or the virtual `discard` target. There is no implicit "forward to next step" — every routing decision must name its destination.

### Boolean Conditions

Boolean expressions (comparisons, `and`/`or`) must use `"true"`/`"false"` as route labels:

```yaml
# CORRECT - boolean condition uses true/false
gates:
  - name: threshold
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": output

# WRONG - boolean condition with non-boolean labels
gates:
  - name: threshold
    condition: "row['amount'] > 1000"
    routes:
      "above": high_values  # ERROR: condition returns True/False, not "above"
      "below": output
```

### Fork Operations

Split rows to multiple parallel paths:

```yaml
gates:
  - name: parallel_analysis
    condition: "True"
    routes:
      "true": fork
    fork_to:
      - sentiment_path
      - entity_path
```

---

## Aggregation Settings

Batch rows until a trigger fires, then process as a group.

```yaml
aggregations:
  - name: batch_stats
    plugin: batch_stats
    input: enriched
    on_success: output
    on_error: discard           # Sink name for batch errors, or 'discard'
    trigger:
      count: 100              # Fire after 100 rows
      timeout_seconds: 3600   # Or after 1 hour
    output_mode: transform
    expected_output_count: 1  # Optional; omit when group_by can emit multiple rows
    options:
      schema:
        mode: observed
      value_field: amount
      group_by: customer_tier  # Optional: one aggregate row per tier
      compute_mean: true
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique aggregation identifier |
| `plugin` | string | **Yes** | Aggregation plugin name |
| `input` | string | **Yes** | Connection name to receive data from |
| `on_success` | string | No | Where successful output rows go (sink name or connection name) |
| `on_error` | string | **Yes** | Sink name for rows that fail batch processing, or `discard` |
| `trigger` | object | **Yes** | When to flush the batch |
| `output_mode` | string | No | `passthrough` or `transform` (default: `transform`) |
| `expected_output_count` | int | No | For `transform` mode: validate output row count |
| `options` | object | No | Plugin-specific configuration |

The `report_assemble` aggregation collates a batch of text rows into a single report row with pagination metadata sourced from the batch context:

```yaml
aggregations:
  - name: pages
    plugin: report_assemble
    input: lines
    on_success: output
    on_error: discard
    trigger:
      count: 80
    output_mode: transform
    expected_output_count: 1
    options:
      schema:
        mode: observed
      text_field: line
      format: markdown
      title: "Run report"
```

Omit `trigger` to emit a single report covering all source rows at end-of-source.

### Trigger Configuration

At least one trigger type is required:

| Trigger | Type | Description |
|---------|------|-------------|
| `count` | int | Fire after N rows accumulated |
| `timeout_seconds` | float | Fire after N seconds since first accept |
| `condition` | string | Fire when expression evaluates to true |

Multiple triggers can be combined (first to fire wins):

```yaml
trigger:
  count: 1000
  timeout_seconds: 3600
  condition: "row['batch_count'] >= 500 and row['batch_age_seconds'] < 30"
```

**Important:** Trigger conditions operate at the **batch level**, not the row level. Only `batch_count` and `batch_age_seconds` are available as row keys. For row-level routing decisions, use Gates instead.

**Note:** End-of-source is always checked implicitly and doesn't need configuration.

### Output Modes

| Mode | Behavior |
|------|----------|
| `transform` | Batch applies transform function to produce results (default) |
| `passthrough` | Batch releases all accepted rows unchanged |

For N→1 aggregation (e.g., computing statistics), use `transform` mode with `expected_output_count: 1` to validate cardinality.

---

## Coalesce Settings

Merge tokens from parallel fork paths back into a single token.

```yaml
coalesce:
  - name: merge_analysis
    branches:
      - sentiment_path
      - entity_path
    policy: require_all
    merge: union

  - name: quorum_merge
    branches:
      - fast_model
      - slow_model
      - fallback_model
    policy: quorum
    quorum_count: 2
    merge: nested
    timeout_seconds: 30
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique coalesce identifier |
| `branches` | list or dict | **Yes** | Branch names to wait for (min 2). List form `[a, b]` is shorthand for `{a: a, b: b}`. Dict form maps branch identity → input connection. |
| `on_success` | string | No | Sink name or connection name for coalesce output (required when coalesce is terminal) |
| `policy` | string | No | How to handle partial arrivals (default: `require_all`) |
| `merge` | string | No | How to combine data (default: `union`) |
| `union_collision_policy` | string | No | Field-level collision resolution for `merge: union` (default: `last_wins`) |
| `timeout_seconds` | float | No | Max wait time |
| `quorum_count` | int | No | Minimum branches required (for `quorum` policy) |
| `select_branch` | string | No | Which branch to take (for `select` merge) |

### Policies

| Policy | Behavior | Requirements |
|--------|----------|--------------|
| `require_all` | Wait for all branches | - |
| `quorum` | Wait for N branches | `quorum_count` required |
| `best_effort` | Wait until timeout, use what arrived | `timeout_seconds` required |
| `first` | Use first branch to arrive | - |

### Merge Strategies

| Strategy | Behavior | Requirements |
|----------|----------|--------------|
| `union` | Combine all fields from all branches | - |
| `nested` | Each branch's data nested under branch name | - |
| `select` | Take data from one specific branch | `select_branch` required |

### Union Collision Policy

When `merge: union` is used and two or more branches emit the same field name, `union_collision_policy` controls how the field-level conflict is resolved. This is **only meaningful for `merge: union`** — it is ignored for `nested` and `select`.

| Value | Behavior |
|-------|----------|
| `last_wins` *(default)* | The last branch in declaration order wins. Matches the historical behavior of union merges. |
| `first_wins` | The first branch in declaration order wins. |
| `fail` | Raise `CoalesceCollisionError` the moment any field collides. No merged row is produced. The full collision record (origin of every field plus each contributing `(branch, value)` pair) is still written to the audit trail on the failed node state. |

> **Note on `fail`:** Collision detection is **name-based**, not value-based. Two branches that both emit a field called `id` with the *same* value still trigger `fail` — the executor does not compare values to decide whether the overlap is "real." If your branches share trivially-identical fields (like an `id` carried unchanged through both transforms), use `last_wins` or `first_wins` instead, or rename the shared fields out of one branch.

Regardless of the chosen value, every union merge records:

- `union_field_origins` — a mapping from every merged field to the branch that produced the winning value. Always populated so auditors can reconstruct field-level provenance even when no collision occurred.
- `union_field_collision_values` — a mapping from each colliding field to the ordered list of `(branch, value)` pairs. Populated only when at least one field collided.

`union_collision_policy` is **orthogonal to `policy`**: `policy` governs branch-level arrival (what to do when some branches never arrive), while `union_collision_policy` governs field-level conflict within an already-assembled merge. They are independent axes and can be combined freely.

```yaml
coalesce:
  - name: strict_merge
    branches:
      - sentiment_path
      - entity_path
    policy: require_all          # branch-level arrival policy
    merge: union
    union_collision_policy: fail  # field-level collision policy — abort on overlap
    on_success: output
```

---

## Pipeline Dependencies

Declare pipelines that must run before this one. Used for multi-pipeline workflows like RAG ingestion (index first, then query).

```yaml
depends_on:
  - name: indexing
    settings: pipelines/index_pipeline.yaml
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique label for this dependency |
| `settings` | string | **Yes** | Path to the dependency pipeline settings file |

Dependencies are executed in order before the main pipeline starts. Each dependency produces a `DependencyRunResult` recorded in the audit trail.

---

## Commencement Gates

Go/no-go conditions evaluated after dependencies complete but before the main pipeline starts. Use these for pre-flight checks (e.g., verifying a vector store is populated).

```yaml
commencement_gates:
  - name: collection_ready
    condition: "probes['products'].document_count > 0"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique label for this gate |
| `condition` | string | **Yes** | Expression evaluated against pre-flight context (including probe results) |

Gate failures raise `CommencementGateFailedError` and abort the pipeline. Gate passes are recorded in the audit trail.

---

## Collection Probes

Vector store readiness checks that run after dependencies and populate context for commencement gates.

```yaml
collection_probes:
  - collection: products
    provider: chroma
    provider_config:
      persist_directory: ./chroma_data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `collection` | string | **Yes** | Collection name to probe |
| `provider` | string | **Yes** | Provider type (e.g., `chroma`) |
| `provider_config` | object | No | Provider-specific connection configuration |

Probe results (document count, metadata) are available to commencement gate expressions via `probes['<collection_name>']`.

---

## Landscape Settings (Audit Trail)

Configure the audit trail database and optional change journal.

```yaml
landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///./runs/audit.db
  export:
    enabled: true
    sink: audit_archive
    format: json
    sign: true
  # Optional: JSONL change journal for emergency backup
  dump_to_jsonl: false
  dump_to_jsonl_path: ./runs/audit.journal.jsonl
  dump_to_jsonl_include_payloads: false
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable audit trail recording |
| `backend` | string | `sqlite` | Database backend: `sqlite`, `sqlcipher`, `postgresql` |
| `encryption_key_env` | string | `ELSPETH_AUDIT_KEY` | Environment variable holding the SQLCipher passphrase (`backend: sqlcipher` only) |
| `url` | string | `sqlite:///./state/audit.db` | SQLAlchemy database URL |
| `export` | object | (disabled) | Post-run audit export configuration |
| `dump_to_jsonl` | bool | `false` | Write append-only JSONL change journal |
| `dump_to_jsonl_path` | string | (derived from url) | Path for JSONL journal file |
| `dump_to_jsonl_fail_on_error` | bool | `false` | Fail the run if journal write fails |
| `dump_to_jsonl_include_payloads` | bool | `false` | Include request/response bodies in journal |
| `dump_to_jsonl_payload_base_path` | string | (from payload_store) | Payload store path for inlining |

### Export Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable audit trail export after run |
| `sink` | string | - | Sink name to export to (required when enabled) |
| `format` | string | `csv` | Export format: `csv`, `json` |
| `sign` | bool | `false` | HMAC sign each record for integrity |

**Audit export signing:** For legal-grade audit trail integrity, enable export signing by setting `landscape.export.sign = true` in your pipeline settings. This produces cryptographically signed exports (HMAC) that can be independently verified. Requires `ELSPETH_SIGNING_KEY` to be set (see [Environment Variables](environment-variables.md)).

### JSONL Change Journal

Enable a redundant JSONL change journal for emergency backup. This is **not** the canonical audit record—use when you need a text-based, append-only backup stream.

```yaml
landscape:
  enabled: true
  url: sqlite:///./runs/audit.db

  # Enable the change journal
  dump_to_jsonl: true
  dump_to_jsonl_path: ./runs/audit.journal.jsonl

  # Include LLM/HTTP request and response bodies
  dump_to_jsonl_include_payloads: true

  # Fail the pipeline if journal writes fail (strict mode)
  dump_to_jsonl_fail_on_error: false
```

**Use cases:**

| Scenario | Recommended Settings |
|----------|---------------------|
| Debugging LLM calls | `dump_to_jsonl: true`, `include_payloads: true` |
| Compliance backup | `dump_to_jsonl: true`, `fail_on_error: true` |
| Production (minimal I/O) | `dump_to_jsonl: false` (default) |

**Notes:**
- Journal is append-only (never modified after write)
- Each line is a self-contained JSON record
- Includes all database commits: rows, transforms, calls, outcomes
- With `include_payloads: true`, LLM prompts and responses are embedded

### PostgreSQL Example

```yaml
landscape:
  backend: postgresql
  url: postgresql://user@host:5432/elspeth
```

**Note:** Passwords in URLs are fingerprinted (not stored) when `ELSPETH_FINGERPRINT_KEY` is set.

---

## Concurrency Settings

Configure parallel processing.

```yaml
concurrency:
  max_workers: 16
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_workers` | int | `4` | Maximum parallel workers |

**Recommendation:** Development: 4, Production: 16

---

## Rate Limit Settings

Limit external API calls to avoid throttling. Rate limits are applied at the **service level** - all plugins using the same service share the rate limit bucket.

```yaml
rate_limit:
  enabled: true
  default_requests_per_minute: 60
  persistence_path: ./rate_limits.db
  services:
    azure_openai:
      requests_per_minute: 100
    azure_content_safety:
      requests_per_minute: 50
    azure_prompt_shield:
      requests_per_minute: 50
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable rate limiting |
| `default_requests_per_minute` | int | `60` | Default per-minute limit for unconfigured services |
| `persistence_path` | string | - | SQLite path for cross-process rate limit state |
| `services` | object | `{}` | Per-service rate limit configurations |

### Service Rate Limit

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `requests_per_minute` | int | **Yes** | Maximum requests per minute for this service |

### Built-in Service Names

ELSPETH's built-in plugins use these service names for rate limiting:

| Service Name | Used By | Description |
|--------------|---------|-------------|
| `azure_openai` | `llm` (provider: azure) | Azure OpenAI API calls |
| `azure_content_safety` | `azure_content_safety` | Azure Content Safety API |
| `azure_prompt_shield` | `azure_prompt_shield` | Azure Prompt Shield API |

**Important:** Service names must use **underscores**, not hyphens (e.g., `azure_openai`, not `azure-openai`). This follows the internal validation pattern `^[a-zA-Z][a-zA-Z0-9_]*`.

### How Rate Limiting Works

1. **Configuration**: Define service limits in your pipeline YAML
2. **Registry**: The `RateLimitRegistry` creates limiters for each configured service
3. **Acquisition**: Plugins acquire rate limit tokens before making external calls
4. **Blocking**: When the limit is reached, calls block until capacity is available

Rate limits apply per-service across all uses in a pipeline. For example, if you have two `llm` transforms (provider: azure), they share the `azure_openai` rate limit.

### Example: Azure LLM Pipeline with Rate Limits

```yaml
source:
  plugin: csv
  on_success: classify_in
  options:
    path: data/prompts.csv
    schema:
      mode: observed

transforms:
  # First LLM transform
  - name: classifier
    plugin: llm
    input: classify_in
    on_success: summarize_in
    on_error: discard
    options:
      provider: azure
      deployment_name: gpt-4o
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      template: "Classify: {{ row.text }}"
      schema:
        mode: observed

  # Second LLM transform - shares rate limit with first
  - name: summarizer
    plugin: llm
    input: summarize_in
    on_success: output
    on_error: discard
    options:
      provider: azure
      deployment_name: gpt-4o
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      template: "Summarize: {{ row.text }}"
      schema:
        mode: observed

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output/results.csv
      schema:
        mode: observed

# Rate limiting - both transforms share this limit
rate_limit:
  enabled: true
  services:
    azure_openai:
      requests_per_minute: 100  # 100 RPM shared across all llm (provider: azure) transforms
```

### Persistence for Distributed Systems

For multi-process or distributed deployments, configure `persistence_path` to share rate limit state:

```yaml
rate_limit:
  enabled: true
  persistence_path: /shared/rate_limits.db  # SQLite file on shared storage
  services:
    azure_openai:
      requests_per_minute: 100
```

This ensures rate limits are respected across multiple pipeline processes hitting the same external APIs.

### Two-Layer Rate Control (LLM Transforms)

LLM transforms like `llm` (provider: azure, with multiple queries) have **two complementary throttling mechanisms** working at different layers:

| Layer | Mechanism | Purpose | When It Acts |
|-------|-----------|---------|--------------|
| **Client** | `RateLimiter` | Proactive prevention | **Before** each API call |
| **Transform** | `PooledExecutor` with AIMD | Reactive handling | **After** receiving 429 |

These are **defense-in-depth**, not competing systems.

**Request Flow:**

```
                          Row arrives
                               │
                               ▼
              ┌────────────────────────────────┐
              │      BatchTransformMixin       │
              │   (row-level pipelining)       │
              └────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      PooledExecutor            │
              │  (query-level parallelism)     │
              │  - Runs N queries in parallel  │
              │  - AIMD retry on 429           │
              └────────────────────────────────┘
                               │
                               ▼  (for each query)
              ┌────────────────────────────────┐
              │      AuditedLLMClient          │
              │  1. _acquire_rate_limit() ◄────┼── Blocks until RPM available
              │  2. Make API call              │
              │  3. Record to audit trail      │
              └────────────────────────────────┘
                               │
                               ▼
                          Azure API
```

**How each layer works:**

1. **RateLimiter (Proactive)** - Configured in YAML under `rate_limit.services`:
   - Blocks each API call until the configured RPM limit has capacity
   - Uses a sliding window (per minute)
   - Shared across all uses of the same service (e.g., `azure_openai`)
   - **Prevents** 429s through smooth, predictable throttling

2. **PooledExecutor AIMD (Reactive)** - Configured in plugin options:
   - Handles 429 errors that slip through (bursts, quota changes, shared quotas)
   - Uses AIMD backoff: multiply delay on 429, subtract on success
   - Retries until `max_capacity_retry_seconds` timeout
   - **Recovers** from 429s gracefully

**Configuration example with both layers:**

```yaml
# Proactive rate limiting (YAML config)
rate_limit:
  enabled: true
  services:
    azure_openai:
      requests_per_minute: 100  # Proactive: block before exceeding this rate

# Reactive handling (plugin options)
transforms:
  - name: multi_query_llm
    plugin: llm
    input: source_out
    on_success: output
    on_error: discard
    options:
      provider: azure
      deployment_name: gpt-4o
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      queries:
        - template: "Classify: {{ row.text }}"
        - template: "Summarize: {{ row.text }}"
      pool_size: 8                      # Concurrent queries per row
      max_dispatch_delay_ms: 5000       # Max AIMD backoff delay (ms)
      max_capacity_retry_seconds: 3600  # Give up after 1 hour of 429s
      # ... other options
```

**Tuning guide:**

| Symptom | Tune This | How |
|---------|-----------|-----|
| Getting 429s frequently | `rate_limit.services.<name>.requests_per_minute` | Lower the RPM |
| Queries too slow (blocking unnecessarily) | `rate_limit.services.<name>.requests_per_minute` | Raise RPM (if quota allows) |
| 429 recovery too aggressive | Plugin `max_dispatch_delay_ms` | Increase max backoff |
| 429s causing row failures | Plugin `max_capacity_retry_seconds` | Increase retry timeout |

**Key insight:** RateLimiter is your first line of defense (smooth, predictable throttling), while PooledExecutor AIMD is your safety net (handles bursts and quota changes gracefully).

---

## Telemetry Settings

Configure operational telemetry exports (OTLP, Azure Monitor, Datadog, console). Telemetry provides **real-time operational visibility** alongside the Landscape audit trail.

**Key distinction:**
- **Landscape**: Legal record, complete lineage, persisted forever, source of truth
- **Telemetry**: Operational visibility, real-time streaming, ephemeral, for dashboards/alerting

```yaml
telemetry:
  enabled: true
  granularity: rows
  backpressure_mode: block
  fail_on_total_exporter_failure: false
  exporters:
    - name: otlp
      options:
        endpoint: http://localhost:4317
        headers:
          Authorization: "Bearer ${OTEL_TOKEN}"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable telemetry emission |
| `granularity` | string | `lifecycle` | Event verbosity: `lifecycle`, `rows`, or `full` |
| `backpressure_mode` | string | `block` | How to handle slow exporters: `block`, `drop`, or `slow` |
| `fail_on_total_exporter_failure` | bool | `true` | Crash run if all exporters fail repeatedly |
| `exporters` | list | `[]` | Exporter configurations |

### Granularity Levels

| Level | Events Emitted | Volume | Use Case |
|-------|----------------|--------|----------|
| `lifecycle` | Run start/complete, phase transitions | ~10-20/run | Production monitoring |
| `rows` | Lifecycle + row creation, transform completion, gate routing, field-resolution mapping | N × M events | Debugging, progress tracking |
| `full` | Rows + external call details (LLM prompts/responses, HTTP, SQL) | High | Deep debugging, call analysis |

**Choosing a granularity:**

```yaml
# Production: minimal overhead, just run lifecycle
telemetry:
  enabled: true
  granularity: lifecycle
  exporters:
    - name: datadog

# Development: see row-by-row progress
telemetry:
  enabled: true
  granularity: rows
  exporters:
    - name: console
      options:
        format: pretty

# Debugging LLM issues: full call details
telemetry:
  enabled: true
  granularity: full
  exporters:
    - name: console
      options:
        format: json
```

### Backpressure Modes

| Mode | Behavior | Trade-off |
|------|----------|-----------|
| `block` | Block pipeline when exporters can't keep up | Complete telemetry, may slow pipeline |
| `drop` | Drop events when buffer is full | No pipeline impact, lossy telemetry |
| `slow` | Adaptive rate limiting | (Not yet implemented) |

**Recommendation:** Use `block` for debugging sessions (complete data), `drop` for production (no pipeline impact).

### Exporter Configuration

Each exporter config has a required name and an `options` block. Exporter-specific keys **must** be placed under `options`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | `console`, `otlp`, `azure_monitor`, `datadog` |
| `options` | object | No | Exporter-specific settings (see below) |

**Secrets convention:** keep non-sensitive values in YAML and secrets in `.env`, referenced with `${VAR}`. For example, `options.endpoint` can be set in YAML while `options.headers.Authorization` comes from `.env`.

### Built-in Exporter Options

**Console**
```yaml
options:
  format: json   # json | pretty
  output: stdout # stdout | stderr
```

**OTLP**
```yaml
options:
  endpoint: http://localhost:4317   # required
  headers:
    Authorization: "Bearer ${OTEL_TOKEN}"  # optional
  batch_size: 100                   # optional
```

**Azure Monitor**
```yaml
options:
  connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}  # required
  batch_size: 100                                              # optional
```

**Datadog**
```yaml
options:
  service_name: "elspeth"         # optional
  env: "production"               # optional
  agent_host: "localhost"         # optional
  agent_port: 8126                # optional
  version: "1.0.0"                # optional
```

### Correlation with Audit Trail

Telemetry events include `run_id` and `token_id` fields that correlate directly with the Landscape audit database. This enables tracing from operational alerts to full lineage investigation.

**Workflow:**

1. **Alert fires** in Datadog/Grafana (e.g., "high error rate on transform X")
2. **Extract `run_id`** from the telemetry event
3. **Investigate with `explain`** command:
   ```bash
   elspeth explain --run <run_id> --database ./runs/audit.db
   ```
4. **Or use the Landscape MCP server** for programmatic access:
   ```bash
   elspeth-mcp --database ./runs/audit.db
   # Then: get_failure_context(run_id)
   ```

**Key correlation fields in telemetry events:**

| Field | Description | Maps To |
|-------|-------------|---------|
| `run_id` | Pipeline execution identifier | `runs.run_id` |
| `token_id` | Row instance identifier | `node_states.token_id` |
| `node_id` | Transform/gate instance | `nodes.node_id` |
| `state_id` | Processing state record | `node_states.state_id` |

---

## Checkpoint Settings

Configure crash recovery checkpointing.

```yaml
checkpoint:
  enabled: true
  frequency: every_n
  checkpoint_interval: 100
  aggregation_boundaries: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable checkpointing |
| `frequency` | string | `every_row` | Checkpoint frequency |
| `checkpoint_interval` | int | - | Row interval (required for `every_n`) |
| `aggregation_boundaries` | bool | `true` | Always checkpoint at aggregation flush |

### Frequency Options

| Frequency | Behavior | Trade-off |
|-----------|----------|-----------|
| `every_row` | Checkpoint after each row | Safest, higher I/O |
| `every_n` | Checkpoint every N rows | Balance safety/performance |
| `aggregation_only` | Checkpoint at aggregation flushes only | Fastest, lose up to batch on crash |

---

## Retry Settings

Configure retry behavior for transient failures.

```yaml
retry:
  max_attempts: 3
  initial_delay_seconds: 1.0
  max_delay_seconds: 60.0
  exponential_base: 2.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_attempts` | int | `3` | Maximum retry attempts |
| `initial_delay_seconds` | float | `1.0` | Initial backoff delay |
| `max_delay_seconds` | float | `60.0` | Maximum backoff delay |
| `exponential_base` | float | `2.0` | Exponential backoff base |

Delay calculation: `min(initial_delay * base^attempt, max_delay)`

---

## Payload Store Settings

Configure storage for large binary payloads.

```yaml
payload_store:
  backend: filesystem
  base_path: .elspeth/payloads
  retention_days: 90
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `filesystem` | Storage backend |
| `base_path` | path | `.elspeth/payloads` | Base path for filesystem backend |
| `retention_days` | int | `90` | Payload retention in days |

---

## Environment Variables

See the [Environment Variables Reference](environment-variables.md) for the complete list of supported environment variables, including:

- **Security variables:** `ELSPETH_FINGERPRINT_KEY`, `ELSPETH_SIGNING_KEY`
- **LLM provider keys:** `OPENROUTER_API_KEY`, `AZURE_OPENAI_API_KEY`
- **Azure service credentials:** Content Safety, Prompt Shield, Blob Storage
- **Telemetry credentials:** `OTEL_TOKEN`, `APPLICATIONINSIGHTS_CONNECTION_STRING`, `DD_API_KEY`
- **Secret field detection patterns**

Configuration is loaded with this precedence (highest first):
1. Environment variables (`ELSPETH_*`)
2. Config file (settings.yaml)
3. Pydantic schema defaults

Nested environment variables use double underscore: `ELSPETH_LANDSCAPE__URL`.

---

## Expression Syntax

Gate conditions and aggregation triggers use a restricted expression language.

### Allowed Constructs

| Construct | Example |
|-----------|---------|
| Field access | `row['field']`, `row.get('field')` |
| Built-in functions | `len()`, `abs()` |
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Boolean operators | `and`, `or`, `not` |
| Arithmetic | `+`, `-`, `*`, `/`, `%` |
| Membership | `in`, `not in` |
| Literals | `True`, `False`, `None`, numbers, strings |
| Ternary | `x if condition else y` |

**`row.get()` does not accept default values.** `row.get('field')` returns `None` if the key is missing. `row.get('field', 'fallback')` is **forbidden** — default values fabricate data the source never provided. Use `row.get('field') is not None` to test for field presence.

### Forbidden Constructs

| Forbidden | Reason |
|-----------|--------|
| Coercive function calls (`int()`, `str()`, `float()`, `bool()`) | Not needed — the source schema guarantees type safety before expressions run |
| Imports | Security |
| Lambda expressions | Security |
| Comprehensions | Security |
| Attribute access (except `row.get()`) | Security |
| F-strings | Security |

### Type Safety

Type coercion functions like `int()` are not needed in expressions. The source schema handles type conversion at the boundary — by the time data reaches a gate or trigger, fields already have the types declared in the schema:

```yaml
source:
  plugin: csv
  options:
    schema:
      fields:
        - "amount: int"  # CSV strings are coerced to int at load time

gates:
  - name: threshold
    condition: "row['amount'] > 1000"  # amount is guaranteed to be int here
```

---

## Complete Example

```yaml
# Source - where data comes from
source:
  plugin: csv
  on_success: raw_data
  options:
    path: data/transactions.csv
    schema:
      mode: fixed
      fields:
        - "id: int"
        - "amount: int"
        - "customer_id: str"
    on_validation_failure: quarantine

# Sinks - where data goes
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output/normal.csv
      schema:
        mode: observed

  high_values:
    plugin: csv
    on_write_failure: discard
    options:
      path: output/high_values.csv
      schema:
        mode: observed

  quarantine:
    plugin: csv
    on_write_failure: discard
    options:
      path: output/quarantine.csv
      schema:
        mode: observed

# Transforms
transforms:
  - name: enricher
    plugin: field_mapper
    input: raw_data
    on_success: enriched
    on_error: quarantine
    options:
      schema:
        mode: observed
      computed:
        processed_at: "row.get('timestamp', 'unknown')"

# Gates - routing decisions
gates:
  - name: amount_threshold
    input: enriched
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": output

# Audit trail
landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///./runs/audit.db
  export:
    enabled: false

# Operational settings
concurrency:
  max_workers: 4

checkpoint:
  enabled: true
  frequency: every_row

retry:
  max_attempts: 3
  initial_delay_seconds: 1.0
  max_delay_seconds: 60.0
  exponential_base: 2.0

rate_limit:
  enabled: true
  default_requests_per_minute: 60
  services:
    azure_openai:
      requests_per_minute: 100

payload_store:
  backend: filesystem
  base_path: .elspeth/payloads
  retention_days: 90

telemetry:
  enabled: true
  granularity: rows
  exporters:
    - name: console
      options:
        format: pretty
    - name: otlp
      options:
        endpoint: http://localhost:4317
        headers:
          Authorization: "Bearer ${OTEL_TOKEN}"
```

---

## See Also

- [Your First Pipeline](../guides/your-first-pipeline.md) - Getting started tutorial
- [Docker Guide](../guides/docker.md) - Container deployment
- [PLUGIN.md](../../PLUGIN.md) - Plugin development
