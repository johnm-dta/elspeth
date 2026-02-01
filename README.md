# ELSPETH

**E**xtensible **L**ayered **S**ecure **P**ipeline **E**ngine for **T**ransformation and **H**andling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Status: RC-2](https://img.shields.io/badge/status-RC--2-yellow.svg)]()

Auditable Sense/Decide/Act pipelines for high-stakes data processing. Every decision traceable to its source.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [The Sense/Decide/Act Model](#the-sensedecideact-model)
- [Example Use Cases](#example-use-cases)
- [Usage](#usage)
  - [Running Pipelines](#running-pipelines)
  - [Explaining Decisions](#explaining-decisions)
  - [Audit Trail Export](#audit-trail-export)
- [ChaosLLM Testing](#chaosllm-testing)
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
- **Plugin Architecture** - Extensible sources, transforms, gates, and sinks via pluggy with dynamic discovery
- **Conditional Routing** - Gates route rows to different sinks based on config-driven expressions (AST-parsed, no eval)
- **Resilient Execution** - Checkpointing for crash recovery, retry logic with backoff, rate limiting, payload retention policies
- **Signed Exports** - HMAC-signed audit exports for legal-grade integrity verification with manifest hash chains
- **LLM Integration** - Azure OpenAI and OpenRouter support with pooled execution, batch processing, and multi-query
- **ChaosLLM Testing** - Fake LLM server for load/stress testing with error injection, latency simulation, and MCP analysis tools

---

## Quick Start

```bash
# Install
git clone https://github.com/johnm-dta/elspeth.git && cd elspeth
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Validate configuration
elspeth validate --settings examples/threshold_gate/settings.yaml

# Run a pipeline
elspeth run --settings examples/threshold_gate/settings.yaml --execute

# Resume an interrupted run
elspeth resume <run_id>
```

See [Your First Pipeline](docs/guides/your-first-pipeline.md) for a complete walkthrough.

---

## The Sense/Decide/Act Model

```
SENSE (Sources)  →  DECIDE (Transforms/Gates)  →  ACT (Sinks)
     │                       │                        │
  Load data            Process & classify       Route to outputs
```

| Stage | What It Does | Examples |
|-------|--------------|----------|
| **Sense** | Load data from any source | CSV, JSON, APIs, databases, message queues |
| **Decide** | Transform and classify rows | LLM query, ML inference, rules engine, threshold gate |
| **Act** | Route to appropriate outputs | File output, database insert, alert webhook, review queue |

**Gates** are special transforms that route rows to different sinks based on classification:

```yaml
gates:
  - name: safety_check
    condition: "row['risk_score'] > 0.8"
    routes:
      "true": high_risk_review
      "false": continue
```

---

## Example Use Cases

| Domain | Sense | Decide | Act |
|--------|-------|--------|-----|
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

## ChaosLLM Testing

ChaosLLM is a fake LLM server for load and stress testing pipelines without real API calls.

**Features:**
- OpenAI + Azure OpenAI compatible chat completion endpoints
- Error injection: rate limits (429), server errors (5xx), timeouts, disconnects, malformed JSON
- Latency simulation and burst patterns for AIMD testing
- Response modes: random, template (Jinja2), echo, preset bank (JSONL)
- SQLite metrics with MCP analysis tools

**Quick start:**

```bash
# Start server with stress testing preset
chaosllm serve --preset=stress_aimd

# Run on custom port with 20% rate limit errors
chaosllm serve --port=9000 --rate-limit-pct=20

# Generate structured JSON with templates
chaosllm serve --response-mode=template

# Analyze metrics
chaosllm-mcp --database ./chaosllm-metrics.db
```

> **Note:** All `chaosllm` commands also work as `elspeth chaosllm`.

See `docs/testing/chaosllm.md` for complete configuration and usage.

---

## Configuration

### Pipeline Configuration

```yaml
# pipeline.yaml
source:
  plugin: csv
  options:
    path: data/input.csv
    schema:
      fields: dynamic

sinks:
  results:
    plugin: csv
    options:
      path: output/results.csv

  flagged:
    plugin: csv
    options:
      path: output/flagged.csv

transforms:
  - plugin: field_mapper
    options:
      schema:
        fields: dynamic
      mappings:
        old_name: new_name

gates:
  - name: quality_gate
    condition: "row['score'] >= 0.7"
    routes:
      "true": continue
      "false": flagged

default_sink: results

landscape:
  url: sqlite:///./audit.db
```

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

This design ensures audit trail integrity while optimizing performance where it matters. See [ADR-001](docs/design/adr/001-plugin-level-concurrency.md) for rationale.

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
  ghcr.io/johnm-dta/elspeth:v0.1.0 \
  run --settings /app/config/pipeline.yaml --execute

# Health check
docker run --rm ghcr.io/johnm-dta/elspeth health --json
```

| Mount | Purpose |
|-------|---------|
| `/app/config` | Pipeline YAML (read-only) |
| `/app/input` | Source data (read-only) |
| `/app/output` | Sink outputs (read-write) |
| `/app/state` | Audit DB, checkpoints (read-write) |

See [Docker Guide](docs/guides/docker.md) for complete deployment documentation.

---

## Architecture

```
elspeth-rapid/
├── src/elspeth/
│   ├── core/           # Config, canonical JSON, DAG, rate limiting, retention
│   │   └── landscape/  # Audit trail (recorder, exporter, schema)
│   ├── contracts/      # Type contracts, schemas, protocol definitions
│   ├── engine/         # Orchestrator, processor, executors, retry
│   ├── plugins/        # Sources, transforms, sinks, LLM integrations
│   ├── tui/            # Terminal UI (Textual)
│   └── cli.py          # Typer CLI
└── tests/
    ├── unit/           # Unit tests
    ├── integration/    # Integration tests
    └── contracts/      # Protocol contract tests
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| CLI | Typer | Commands: run, explain, validate, resume, purge |
| TUI | Textual | Interactive lineage explorer |
| Config | Dynaconf + Pydantic | Multi-source with env var expansion |
| Plugins | pluggy | Dynamic discovery, extensible components |
| Audit | SQLAlchemy Core | SQLite (dev) / PostgreSQL (prod) |
| Canonical | RFC 8785 (JCS) | Deterministic JSON hashing |
| LLM | Azure OpenAI + OpenRouter | Direct integration with pooled execution |
| Templates | Jinja2 | Prompt templating and path generation |

See [Architecture Documentation](ARCHITECTURE.md) for C4 diagrams and detailed design.

---

## Documentation

| Document | Audience | Content |
|----------|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developers | C4 diagrams, data flows, component details |
| [PLUGIN.md](PLUGIN.md) | Plugin Authors | How to create sources, transforms, sinks |
| [docs/design/requirements.md](docs/design/requirements.md) | All | 323 verified requirements with implementation status |
| [docs/design/adr/](docs/design/adr/) | Architects | Architecture Decision Records (why we built it this way) |
| [CLAUDE.md](CLAUDE.md) | AI Assistants | Project context, trust model, patterns |
| [docs/guides/](docs/guides/) | All | Tutorials and how-to guides |
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
|-------------|------------------|
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

# Azure Blob integration tests (Azurite emulator)
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
