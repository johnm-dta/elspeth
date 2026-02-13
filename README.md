# ELSPETH

**E**xtensible **L**ayered **S**ecure **P**ipeline **E**ngine for **T**ransformation and **H**andling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
![Status: RC-3](https://img.shields.io/badge/status-RC--3-yellow.svg)

Auditable Sense/Decide/Act pipelines for high-stakes data processing. Every decision traceable to its source.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [The Sense/Decide/Act Model](#the-sensedecideact-model)
- [Data Trust Model](#data-trust-model)
- [Example Use Cases](#example-use-cases)
- [Usage](#usage)
  - [Running Pipelines](#running-pipelines)
  - [Explaining Decisions](#explaining-decisions)
  - [Audit Trail Export](#audit-trail-export)
- [Chaos Testing](#chaos-testing)
  - [ChaosLLM](#chaosllm)
  - [ChaosWeb](#chaosweb)
- [Landscape MCP Server](#landscape-mcp-server)
- [Configuration](#configuration)
- [Docker](#docker)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [When to Use ELSPETH](#when-to-use-elspeth)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Complete Audit Trail** - Every transform, every routing decision, every external call recorded with payload storage
- **Explain Any Decision** - `elspeth explain --run latest --row 42 --database <path/to/audit.db>` launches TUI to explore why any row reached its destination
- **Declarative DAG Wiring** - Every edge is explicitly named and validated at construction time — no implicit routing
- **Conditional Routing** - Gates route rows to different sinks based on config-driven expressions (AST-parsed, no eval)
- **Plugin Architecture** - Extensible sources, transforms, and sinks via pluggy with dynamic discovery
- **Encryption at Rest** - SQLCipher encryption for the Landscape audit database via passphrase or environment variable
- **Resilient Execution** - Checkpointing for crash recovery, retry logic with backoff, rate limiting, payload retention policies
- **Signed Exports** - HMAC-signed audit exports for legal-grade integrity verification with manifest hash chains
- **LLM Integration** - Azure OpenAI and OpenRouter support with pooled execution, batch processing, and multi-query
- **Landscape MCP Server** - Read-only MCP analysis server for debugging pipeline failures against the audit database
- **Chaos Testing** - ChaosLLM (fake LLM) and ChaosWeb (fake web server) for load/stress testing with error injection, latency simulation, and metrics

---

## Quick Start

```bash
# Install
git clone https://github.com/johnm-dta/elspeth.git && cd elspeth
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Validate configuration
elspeth validate --settings examples/threshold_gate/settings.yaml

# Run a pipeline (audit DB created at the path in settings.yaml landscape.url)
elspeth run --settings examples/threshold_gate/settings.yaml --execute

# Explore why a row reached its destination
elspeth explain --run <run_id> --row <row_id> \
  --database examples/threshold_gate/runs/audit.db

# Resume an interrupted run
elspeth resume <run_id>
```

See [Your First Pipeline](docs/guides/your-first-pipeline.md) for a complete walkthrough.

---

## The Sense/Decide/Act Model

```text
SENSE (Sources)  →  DECIDE (Transforms/Gates)  →  ACT (Sinks)
     │                       │                        │
  Load data            Process & classify       Route to outputs
```

| Stage | What It Does | Examples |
| ----- | ------------ | -------- |
| **Sense** | Load data from any source | CSV, JSON, APIs, databases, message queues |
| **Decide** | Transform and classify rows | LLM query, ML inference, rules engine, threshold gate |
| **Act** | Route to appropriate outputs | File output, database insert, alert webhook, review queue |

**Gates** route rows to different sinks based on config-driven expressions:

```yaml
gates:
- name: safety_check
  input: processed          # Explicit input connection
  condition: "row['risk_score'] > 0.8"
  routes:
    "true": high_risk_review  # Route to named sink
    "false": approved         # Route to named sink
```

Every edge in the DAG is explicitly declared — no implicit routing conventions.

---

## Data Trust Model

ELSPETH enforces a **three-tier trust model** that governs how data is handled at every stage of the pipeline:

| Tier | Data Source | Trust Level | Error Strategy |
| ---- | ----------- | ----------- | -------------- |
| **Tier 1** | Audit database (Landscape) | Full trust | Crash on any anomaly — bad audit data means corruption or tampering |
| **Tier 2** | Pipeline data (post-source) | Elevated trust | Types are valid (source validated them); wrap operations on row values |
| **Tier 3** | External input (sources, API responses) | Zero trust | Validate at boundary, coerce where possible, quarantine failures |

**Coercion is only allowed at trust boundaries:** sources ingesting external data, and transforms receiving LLM/API responses. Once data enters the pipeline with valid types, downstream transforms trust those types. Wrong types downstream are upstream bugs to fix, not data quality issues to handle gracefully.

This means a CSV with garbage in row 500 won't crash your 10,000-row pipeline (Tier 3: quarantine the row, keep processing). But a corrupted audit record will crash immediately (Tier 1: the audit trail is the legal record, and silently coercing bad data would be evidence tampering).

```text
EXTERNAL DATA              PIPELINE DATA              AUDIT TRAIL
(zero trust)               (elevated trust)           (full trust)

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ External Source │        │ Transform/Sink  │        │ Landscape DB    │
│                 │        │                 │        │                 │
│ • Coerce OK     │───────►│ • No coercion   │───────►│ • Crash on      │
│ • Validate      │ types  │ • Expect types  │ record │   any anomaly   │
│ • Quarantine    │ valid  │ • Wrap ops on   │ what   │ • No coercion   │
│   failures      │        │   row values    │ we saw │   ever          │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

See [Data Trust and Error Handling](docs/guides/data-trust-and-error-handling.md) for the complete model with code examples.

---

## Example Use Cases

| Domain | Sense | Decide | Act |
| ------ | ----- | ------ | --- |
| **Tender Evaluation** | CSV of submissions | LLM classification + safety gates | Results CSV, abuse review queue |
| **Weather Monitoring** | Sensor API feed | Threshold + ML anomaly detection | Routine log, warning, emergency alert |
| **Satellite Operations** | Telemetry stream | Anomaly classifier | Routine log, investigation ticket |
| **Financial Compliance** | Transaction feed | Rules engine + ML fraud detection | Approved, flagged, blocked |
| **Content Moderation** | User submissions | Safety classifier | Published, human review, rejected |

Same framework. Different plugins. Full audit trail.

---

## Usage

### Running Pipelines

```bash
# Validate configuration before running
elspeth validate --settings pipeline.yaml

# Execute a pipeline
elspeth run --settings pipeline.yaml --execute

# Resume an interrupted run (run_id is positional)
elspeth resume abc123

# List available plugins
elspeth plugins list
```

### Explaining Decisions

ELSPETH records complete lineage for every row. The audit database captures:

- Source row with content hash
- Every transform applied (input/output hashes)
- Every gate evaluation with condition result
- Final destination and artifact hash

```bash
# Launch lineage explorer TUI (database path required)
elspeth explain --run <run_id> --row <row_id> --database <path/to/audit.db>
```

For programmatic access, query the Landscape database directly using the `LandscapeRecorder` API.

### Audit Trail Export

Export the complete audit trail for compliance and legal inquiry:

```yaml
landscape:
  export:
    enabled: true
    sink: audit_archive
    format: json
    sign: true  # HMAC signature per record
```

```bash
export ELSPETH_SIGNING_KEY="your-secret-key"
elspeth run --settings pipeline.yaml --execute
```

Signed exports include:

- Every record with HMAC-SHA256 signature
- Manifest with total count and running hash
- Timestamp for chain-of-custody verification

### JSONL Change Journal (Optional)

Enable a redundant JSONL change journal to record committed database writes as
an emergency backup. **Disabled by default.** This is not the canonical audit
record—use only when you need a text-based, append-only backup stream.

```yaml
landscape:
  dump_to_jsonl: true
  dump_to_jsonl_path: ./runs/audit.journal.jsonl
  # Include request/response payloads for LLM/HTTP calls
  dump_to_jsonl_include_payloads: true
```

---

## Chaos Testing

ELSPETH includes chaos testing servers for load and stress testing without real external calls.

### ChaosLLM

Fake LLM server compatible with OpenAI and Azure OpenAI chat completion endpoints.

- Error injection: rate limits (429), server errors (5xx), timeouts, disconnects, malformed JSON
- Latency simulation and burst patterns for AIMD testing
- Response modes: random, template (Jinja2), echo, preset bank (JSONL)
- SQLite metrics with MCP analysis tools

```bash
# Start server with stress testing preset
chaosllm serve --preset=stress_aimd

# Run on custom port with 20% rate limit errors
chaosllm serve --port=9000 --rate-limit-pct=20

# Analyze metrics
chaosllm-mcp --database ./chaosllm-metrics.db
```

> **Note:** All `chaosllm` commands also work as `elspeth chaosllm`.

See `docs/reference/chaosllm.md` for complete configuration and usage.

### ChaosWeb

Fake web server for testing the `web_scrape` transform with configurable failure modes.

- HTTP error injection: 4xx/5xx codes, timeouts, connection resets, slow responses
- Content generation: HTML pages with configurable structure and encoding
- Preset profiles: `gentle`, `realistic`, `silent`, `stress_scraping`, `stress_extreme`
- Request metrics and failure tracking

```bash
# Start with realistic failure profile
chaosweb serve --preset=realistic

# Custom error rate and port
chaosweb serve --port=9001 --error-rate=30

# Use in pipeline settings
# source.options.base_url: http://localhost:9001
```

> **Note:** All `chaosweb` commands also work as `elspeth chaosweb`.

See `examples/chaosweb/` for a complete pipeline example.

---

## Landscape MCP Server

A read-only MCP server for debugging pipeline failures against the audit database:

```bash
# Auto-discover databases
elspeth-mcp

# Explicit database path
elspeth-mcp --database sqlite:///./runs/audit.db

# Encrypted database
ELSPETH_AUDIT_PASSPHRASE="secret" elspeth-mcp --database sqlite:///./runs/audit.db
```

Key tools: `diagnose()` (what's broken?), `get_failure_context(run_id)` (deep dive), `explain_token(run_id, token_id)` (row lineage), `get_performance_report(run_id)` (bottlenecks).

See `docs/guides/landscape-mcp-analysis.md` for the full tool reference.

---

## Configuration

### Pipeline Configuration

```yaml
# pipeline.yaml
source:
  plugin: csv
  on_success: validated       # Named output connection
  options:
    path: data/input.csv
    schema:
      fields: dynamic

transforms:
- name: enrich
  plugin: field_mapper
  input: validated            # Connects to source output
  on_success: enriched        # Named output connection
  options:
    schema:
      fields: dynamic
    mapping:
      old_name: new_name

gates:
- name: quality_gate
  input: enriched             # Connects to transform output
  condition: "row['score'] >= 0.7"
  routes:
    "true": results           # Route to named sink
    "false": flagged          # Route to named sink

sinks:
  results:
    plugin: csv
    options:
      path: output/results.csv
  flagged:
    plugin: csv
    options:
      path: output/flagged.csv

landscape:
  url: sqlite:///./audit.db
```

### Field Normalization

ELSPETH handles messy external headers through source-side normalization and sink-side display restoration.

#### Source Normalization

Normalize messy headers (e.g., `"User ID"`, `"CaSE Study1 !!!! xx!"`) to valid Python identifiers at the source boundary:

```yaml
source:
  plugin: csv
  options:
    path: data/input.csv

    # Auto-normalize messy headers to valid Python identifiers
    normalize_fields: true  # "User ID" → "user_id"

    # Optional: Override specific normalized names
    field_mapping:
      case_study1_xx: cs1  # After normalization, rename to cs1
```

#### Sink Display Headers

Restore original header names in output files while keeping the internal data layer clean:

```yaml
sinks:
  output:
    plugin: csv
    options:
      path: output/results.csv

      # Option 1: Explicit mapping (full control)
      display_headers:
        user_id: "User ID"
        amount: "Transaction Amount"

      # Option 2: Auto-restore from source (convenience)
      # restore_source_headers: true
```

| Option                   | Use When                                                          |
| ------------------------ | ----------------------------------------------------------------- |
| `display_headers`        | You need custom output names or don't want source coupling        |
| `restore_source_headers` | You want to restore exact original headers from normalized source |

Transform-added fields (not in source) use their normalized names when restoring.

### Environment Variables

```bash
# Required for production (secret fingerprinting)
export ELSPETH_FINGERPRINT_KEY="your-stable-key"

# Azure Key Vault (alternative to direct key)
export ELSPETH_KEYVAULT_URL="https://your-vault.vault.azure.net/"
export ELSPETH_KEYVAULT_SECRET_NAME="elspeth-fingerprint-key"

# LLM API keys
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export OPENROUTER_API_KEY="sk-or-..."

# Signed exports
export ELSPETH_SIGNING_KEY="your-signing-key"

# Audit database encryption (SQLCipher)
export ELSPETH_AUDIT_PASSPHRASE="your-audit-passphrase"
```

ELSPETH automatically loads `.env` files. Use `--no-dotenv` to skip in CI/CD.

<details>
<summary><strong>Advanced Configuration</strong></summary>

### Hierarchical Settings

```yaml
# Base defaults
default:
  concurrency:
    max_workers: 4

# Profile overrides
profiles:
  production:
    concurrency:
      max_workers: 16
    landscape:
      url: postgresql://...
```

```bash
elspeth run --settings config.yaml --profile production
```

### Secret Fingerprinting

ELSPETH fingerprints secrets before storing in the audit trail:

- **Production**: Set `ELSPETH_FINGERPRINT_KEY` (stable key for fingerprint consistency)
- **Development**: Set `ELSPETH_ALLOW_RAW_SECRETS=true` (redacts instead of fingerprints)

Missing fingerprint key with secrets in config causes startup failure (fail-closed design).

### Payload Store

Large blobs are stored separately from the audit database:

```yaml
payload_store:
  base_path: ./state/payloads
  retention_days: 90
```

### Concurrency Model

ELSPETH uses **plugin-level concurrency** rather than orchestrator-level parallelism:

- **Orchestrator**: Single-threaded, sequential token processing (deterministic audit trail)
- **Plugins**: Internally parallelize I/O-bound operations (LLM batching, DB bulk writes)

```yaml
concurrency:
  max_workers: 4  # Available for plugin use (e.g., LLM thread pools)
```

This design ensures audit trail integrity while optimizing performance where it matters. See [ADR-001](docs/architecture/adr/001-plugin-level-concurrency.md) for rationale.

### Rate Limiting

Control external API call rates to avoid provider throttling:

```yaml
rate_limit:
  enabled: true
  services:
    azure_openai:           # Azure OpenAI LLM transforms
      requests_per_minute: 100
    azure_content_safety:   # Content Safety transform
      requests_per_minute: 50
    azure_prompt_shield:    # Prompt Shield transform
      requests_per_minute: 50
```

Rate limits are **per-service** - all plugins using the same service share the bucket. See [Configuration Reference](docs/reference/configuration.md#rate-limit-settings) for details.

</details>

---

## Docker

ELSPETH is available as a Docker container:

```bash
# Run a pipeline
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:v0.3.0 \
  run --settings /app/config/pipeline.yaml --execute

# Health check
docker run --rm ghcr.io/johnm-dta/elspeth health --json
```

| Mount | Purpose |
| ----- | ------- |
| `/app/config` | Pipeline YAML (read-only) |
| `/app/input` | Source data (read-only) |
| `/app/output` | Sink outputs (read-write) |
| `/app/state` | Audit DB, checkpoints (read-write) |

See [Docker Guide](docs/guides/docker.md) for complete deployment documentation.

---

## Architecture

```text
elspeth/
├── src/elspeth/
│   ├── core/               # Config, canonical JSON, rate limiting, retention
│   │   ├── dag/            # DAG construction, validation, graph models (NetworkX)
│   │   └── landscape/      # Audit trail (recorder, exporter, schema, SQLCipher)
│   ├── contracts/          # Type contracts, schemas, protocol definitions
│   ├── engine/             # Orchestrator, processor, retry, DAG navigator
│   │   └── executors/      # Transform, gate, sink, aggregation executors
│   ├── plugins/            # Sources, transforms, sinks, LLM integrations
│   ├── mcp/                # Landscape MCP analysis server
│   ├── testing/            # ChaosLLM, ChaosWeb, ChaosEngine test servers
│   ├── tui/                # Terminal UI (Textual)
│   └── cli.py              # Typer CLI
└── tests/
    ├── unit/               # Unit tests
    ├── integration/        # Integration tests
    ├── property/           # Hypothesis property-based tests
    ├── e2e/                # End-to-end pipeline tests
    └── performance/        # Benchmarks, stress, scalability tests
```

| Component | Technology | Purpose |
| --------- | ---------- | ------- |
| CLI | Typer | Commands: run, explain, validate, resume, purge |
| TUI | Textual | Interactive lineage explorer |
| Config | Dynaconf + Pydantic | Multi-source with env var expansion |
| Plugins | pluggy | Dynamic discovery, extensible components |
| Audit | SQLAlchemy Core | SQLite/SQLCipher (dev) / PostgreSQL (prod) |
| MCP | Landscape MCP Server | Read-only audit database analysis and debugging |
| Canonical | RFC 8785 (JCS) | Deterministic JSON hashing |
| DAG | NetworkX | Graph validation, topological sort, cycle detection |
| LLM | Azure OpenAI + OpenRouter | Direct integration with pooled execution |
| Templates | Jinja2 | Prompt templating and path generation |

See [Architecture Documentation](ARCHITECTURE.md) for C4 diagrams and detailed design.

---

## Documentation

| Document | Audience | Content |
| -------- | -------- | ------- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developers | C4 diagrams, data flows, component details |
| [PLUGIN.md](PLUGIN.md) | Plugin Authors | How to create sources, transforms, sinks |
| [docs/architecture/requirements.md](docs/architecture/requirements.md) | All | 323 verified requirements with implementation status |
| [docs/architecture/adr/](docs/architecture/adr/) | Architects | Architecture Decision Records (5 ADRs including sink routing and DAG wiring) |
| [CLAUDE.md](CLAUDE.md) | AI Assistants | Project context, trust model, patterns |
| [docs/guides/](docs/guides/) | All | Tutorials, MCP analysis guide, data trust model |
| [docs/reference/](docs/reference/) | Developers | Configuration reference |
| [docs/runbooks/](docs/runbooks/) | Operators | Deployment and operations |

---

## When to Use ELSPETH

### Good Fit

- Decisions that need to be explainable to auditors
- Regulatory or compliance requirements
- Systems where "why did it do that?" matters
- Workflows mixing automated and human review
- High-stakes processing with legal accountability

### Consider Alternatives

| If You Need | Consider Instead |
| ----------- | ---------------- |
| High-throughput ETL | Spark, dbt |
| Sub-second streaming | Flink, Kafka Streams |
| Simple scripts, no audit | Plain Python |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Development setup:**

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,azure]"

# Install Azurite (Azure Blob Storage emulator for integration tests)
npm install

# Run tests
.venv/bin/python -m pytest tests/ -v

# Type checking
.venv/bin/python -m mypy src/

# Linting
.venv/bin/python -m ruff check src/
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

Built for systems where decisions must be **traceable, reliable, and defensible**.
