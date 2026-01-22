# Configuration Reference

Complete reference for ELSPETH pipeline configuration.

---

## Table of Contents

- [Configuration File Format](#configuration-file-format)
- [Top-Level Settings](#top-level-settings)
- [Datasource Settings](#datasource-settings)
- [Sink Settings](#sink-settings)
- [Transform Settings (row_plugins)](#transform-settings-row_plugins)
- [Gate Settings](#gate-settings)
- [Aggregation Settings](#aggregation-settings)
- [Coalesce Settings](#coalesce-settings)
- [Landscape Settings (Audit Trail)](#landscape-settings-audit-trail)
- [Concurrency Settings](#concurrency-settings)
- [Rate Limit Settings](#rate-limit-settings)
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
| `datasource` | object | **Yes** | - | Source plugin configuration (exactly one per run) |
| `sinks` | object | **Yes** | - | Named sink configurations (at least one required) |
| `output_sink` | string | **Yes** | - | Default sink for rows completing the pipeline |
| `run_mode` | string | No | `"live"` | Execution mode: `live`, `replay`, `verify` |
| `replay_source_run_id` | string | No | - | Run ID to replay/verify against (required for replay/verify modes) |
| `row_plugins` | list | No | `[]` | Ordered transforms to apply |
| `gates` | list | No | `[]` | Config-driven routing gates |
| `coalesce` | list | No | `[]` | Fork path merge configurations |
| `aggregations` | list | No | `[]` | Batch processing configurations |
| `landscape` | object | No | (defaults) | Audit trail configuration |
| `concurrency` | object | No | (defaults) | Parallel processing settings |
| `retry` | object | No | (defaults) | Retry behavior settings |
| `payload_store` | object | No | (defaults) | Large blob storage settings |
| `checkpoint` | object | No | (defaults) | Crash recovery settings |
| `rate_limit` | object | No | (defaults) | External call rate limiting |

### Run Modes

| Mode | Behavior |
|------|----------|
| `live` | Execute normally, make real external calls |
| `replay` | Use recorded responses from a previous run |
| `verify` | Compare new results against a previous run |

---

## Datasource Settings

Configures the single data source for the pipeline.

```yaml
datasource:
  plugin: csv
  options:
    path: data/input.csv
    schema:
      mode: free
      fields:
        - "id: int"
        - "amount: int"
    on_validation_failure: quarantine
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plugin` | string | **Yes** | Plugin name: `csv`, `json`, `null` |
| `options` | object | No | Plugin-specific configuration |

### Available Source Plugins

| Plugin | Purpose |
|--------|---------|
| `csv` | Load from CSV file |
| `json` | Load from JSON file or JSONL |
| `null` | Empty source (for testing) |

### Schema Options

```yaml
schema:
  mode: free          # free, strict, or dynamic
  fields:
    - "id: int"       # Field name and type
    - "name: str"
    - "amount: float"
on_validation_failure: quarantine  # quarantine or discard
```

| Schema Mode | Behavior |
|-------------|----------|
| `free` | Accept any fields, coerce specified types |
| `strict` | Require exactly the specified fields |
| `dynamic` | Infer schema from first row |

---

## Sink Settings

Named output destinations. At least one required.

```yaml
sinks:
  output:
    plugin: csv
    options:
      path: output/results.csv
      schema:
        fields: dynamic

  flagged:
    plugin: csv
    options:
      path: output/flagged.csv
      schema:
        fields: dynamic

  quarantine:
    plugin: csv
    options:
      path: output/quarantine.csv
      schema:
        fields: dynamic

output_sink: output  # Default destination
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plugin` | string | **Yes** | Plugin name: `csv`, `json`, `null` |
| `options` | object | No | Plugin-specific configuration |

### Available Sink Plugins

| Plugin | Purpose |
|--------|---------|
| `csv` | Write to CSV file |
| `json` | Write to JSON file |
| `null` | Discard output (for testing) |

---

## Transform Settings (row_plugins)

Ordered list of transforms applied to each row.

```yaml
row_plugins:
  - plugin: field_mapper
    options:
      schema:
        fields: dynamic
      mappings:
        old_field: new_field
      computed:
        full_name: "row['first_name'] + ' ' + row['last_name']"

  - plugin: passthrough
    options: {}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plugin` | string | **Yes** | Plugin name |
| `options` | object | No | Plugin-specific configuration |

### Available Transform Plugins

| Plugin | Purpose |
|--------|---------|
| `passthrough` | Pass rows unchanged |
| `field_mapper` | Rename, compute, drop fields |

---

## Gate Settings

Config-driven routing based on expressions. Gates evaluate conditions and route rows to sinks or forward them.

```yaml
gates:
  - name: quality_check
    condition: "row['confidence'] >= 0.85"
    routes:
      "true": continue      # Forward to next step
      "false": review_sink  # Route to named sink

  - name: amount_threshold
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": continue
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique gate identifier |
| `condition` | string | **Yes** | Expression to evaluate (see [Expression Syntax](#expression-syntax)) |
| `routes` | object | **Yes** | Maps evaluation results to destinations |
| `fork_to` | list | No | Branch paths for fork operations |

### Route Destinations

| Destination | Behavior |
|-------------|----------|
| `continue` | Forward to next pipeline step |
| `<sink_name>` | Route directly to named sink |
| `fork` | Split to multiple paths (requires `fork_to`) |

### Boolean Conditions

Boolean expressions (comparisons, `and`/`or`) must use `"true"`/`"false"` as route labels:

```yaml
# CORRECT - boolean condition uses true/false
gates:
  - name: threshold
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": continue

# WRONG - boolean condition with non-boolean labels
gates:
  - name: threshold
    condition: "row['amount'] > 1000"
    routes:
      "above": high_values  # ERROR: condition returns True/False, not "above"
      "below": continue
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

**Note:** The label `continue` is reserved and cannot be used as a route label or fork branch name.

---

## Aggregation Settings

Batch rows until a trigger fires, then process as a group.

```yaml
aggregations:
  - name: batch_stats
    plugin: stats_aggregation
    trigger:
      count: 100              # Fire after 100 rows
      timeout_seconds: 3600   # Or after 1 hour
    output_mode: single
    options:
      fields: ["value"]
      compute_mean: true
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique aggregation identifier |
| `plugin` | string | **Yes** | Aggregation plugin name |
| `trigger` | object | **Yes** | When to flush the batch |
| `output_mode` | string | No | `single`, `passthrough`, `transform` (default: `single`) |
| `options` | object | No | Plugin-specific configuration |

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
  condition: "row['type'] == 'flush_signal'"
```

**Note:** End-of-source is always checked implicitly and doesn't need configuration.

### Output Modes

| Mode | Behavior |
|------|----------|
| `single` | Batch produces one aggregated result row |
| `passthrough` | Batch releases all accepted rows unchanged |
| `transform` | Batch applies transform function to produce results |

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
| `branches` | list | **Yes** | Branch names to wait for (min 2) |
| `policy` | string | No | How to handle partial arrivals (default: `require_all`) |
| `merge` | string | No | How to combine data (default: `union`) |
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

---

## Landscape Settings (Audit Trail)

Configure the audit trail database.

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
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable audit trail recording |
| `backend` | string | `sqlite` | Database backend: `sqlite`, `postgresql` |
| `url` | string | `sqlite:///./runs/audit.db` | SQLAlchemy database URL |
| `export` | object | (disabled) | Post-run export configuration |

### Export Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable audit trail export after run |
| `sink` | string | - | Sink name to export to (required when enabled) |
| `format` | string | `csv` | Export format: `csv`, `json` |
| `sign` | bool | `false` | HMAC sign each record for integrity |

### PostgreSQL Example

```yaml
landscape:
  backend: postgresql
  url: postgresql://user:password@host:5432/elspeth
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

Limit external API calls to avoid throttling.

```yaml
rate_limit:
  enabled: true
  default_requests_per_second: 10
  default_requests_per_minute: 100
  persistence_path: ./rate_limits.db
  services:
    openai:
      requests_per_second: 5
      requests_per_minute: 100
    weather_api:
      requests_per_second: 20
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable rate limiting |
| `default_requests_per_second` | int | `10` | Default rate limit |
| `default_requests_per_minute` | int | - | Optional per-minute limit |
| `persistence_path` | string | - | SQLite path for cross-process limits |
| `services` | object | `{}` | Per-service configurations |

### Service Rate Limit

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `requests_per_second` | int | **Yes** | Maximum requests per second |
| `requests_per_minute` | int | No | Maximum requests per minute |

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

### Required Variables

| Variable | Purpose | When Required |
|----------|---------|---------------|
| `ELSPETH_FINGERPRINT_KEY` | Secret fingerprinting | Config contains API keys or passwords |
| `ELSPETH_SIGNING_KEY` | Signed audit exports | `landscape.export.sign: true` |

### Optional Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ELSPETH_ALLOW_RAW_SECRETS` | Skip fingerprinting (dev only) | `false` |

### Secret Field Detection

Fields with these names are automatically fingerprinted:
- Exact matches: `api_key`, `token`, `password`, `secret`, `credential`
- Suffixes: `_secret`, `_key`, `_token`, `_password`, `_credential`

---

## Expression Syntax

Gate conditions and aggregation triggers use a restricted expression language.

### Allowed Constructs

| Construct | Example |
|-----------|---------|
| Field access | `row['field']`, `row.get('field', default)` |
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Boolean operators | `and`, `or`, `not` |
| Arithmetic | `+`, `-`, `*`, `/`, `%` |
| Membership | `in`, `not in` |
| Literals | `True`, `False`, `None`, numbers, strings |
| Ternary | `x if condition else y` |

### Forbidden Constructs

| Forbidden | Reason |
|-----------|--------|
| Function calls (`int()`, `len()`, `str()`) | Security - no arbitrary function execution |
| Imports | Security |
| Lambda expressions | Security |
| Comprehensions | Security |
| Attribute access (except `row.get`) | Security |
| F-strings | Security |

### Type Coercion

The expression parser does **not** allow type coercion functions. Instead, coerce types at the source:

```yaml
# CORRECT - coerce at source
datasource:
  plugin: csv
  options:
    schema:
      fields:
        - "amount: int"  # CSV strings coerced to int

gates:
  - name: threshold
    condition: "row['amount'] > 1000"  # amount is already int

# WRONG - function calls not allowed
gates:
  - name: threshold
    condition: "int(row['amount']) > 1000"  # ERROR: int() forbidden
```

---

## Complete Example

```yaml
# Datasource - where data comes from
datasource:
  plugin: csv
  options:
    path: data/transactions.csv
    schema:
      mode: free
      fields:
        - "id: int"
        - "amount: int"
        - "customer_id: str"
    on_validation_failure: quarantine

# Sinks - where data goes
sinks:
  output:
    plugin: csv
    options:
      path: output/normal.csv
      schema:
        fields: dynamic

  high_values:
    plugin: csv
    options:
      path: output/high_values.csv
      schema:
        fields: dynamic

  quarantine:
    plugin: csv
    options:
      path: output/quarantine.csv
      schema:
        fields: dynamic

output_sink: output

# Transforms
row_plugins:
  - plugin: field_mapper
    options:
      schema:
        fields: dynamic
      computed:
        processed_at: "row.get('timestamp', 'unknown')"

# Gates - routing decisions
gates:
  - name: amount_threshold
    condition: "row['amount'] > 1000"
    routes:
      "true": high_values
      "false": continue

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
  default_requests_per_second: 10

payload_store:
  backend: filesystem
  base_path: .elspeth/payloads
  retention_days: 90
```

---

## See Also

- [Your First Pipeline](../guides/your-first-pipeline.md) - Getting started tutorial
- [Docker Guide](../guides/docker.md) - Container deployment
- [PLUGIN.md](../../PLUGIN.md) - Plugin development
