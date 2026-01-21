# ELSPETH

**E**xtensible **L**ayered **S**ecure **P**ipeline **E**ngine for **T**ransformation and **H**andling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Status: RC-1](https://img.shields.io/badge/status-RC--1-yellow.svg)]()

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
- [Configuration](#configuration)
- [Docker](#docker)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [When to Use ELSPETH](#when-to-use-elspeth)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Complete Audit Trail** - Every transform, every routing decision, every external call recorded
- **Explain Any Decision** - `elspeth explain --row 42` shows exactly why any row reached its destination
- **Plugin Architecture** - Extensible sources, transforms, gates, and sinks via pluggy
- **Conditional Routing** - Gates route rows to different sinks based on classification
- **Production Ready** - Checkpointing, retry logic, rate limiting, concurrent processing
- **Signed Exports** - HMAC-signed audit exports for legal-grade integrity verification
- **LLM Integration** - Built-in support for 100+ LLM providers via LiteLLM (Phase 6)

---

## Quick Start

```bash
# Install
git clone https://github.com/johnm-dta/elspeth.git && cd elspeth
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run a pipeline
elspeth run --settings examples/threshold_gate/settings.yaml --execute

# Explore lineage (launches TUI)
elspeth explain --run latest --row 2
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

ELSPETH records complete lineage for every row. The `explain` command launches an interactive TUI to explore lineage:

```bash
# Launch lineage explorer TUI
elspeth explain --run latest --row 2

# Text output (coming soon - TUI is currently the only interface)
# elspeth explain --run latest --row 2 --no-tui
```

The audit database records:
- Source row with content hash
- Every transform applied (input/output hashes)
- Every gate evaluation with condition result
- Final destination and artifact hash

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

---

## Configuration

### Pipeline Configuration

```yaml
# pipeline.yaml
datasource:
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

row_plugins:
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

output_sink: results

landscape:
  url: sqlite:///./audit.db
```

### Environment Variables

```bash
# Required for production (secret fingerprinting)
export ELSPETH_FINGERPRINT_KEY="your-stable-key"

# LLM API keys
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
  retention:
    max_age_days: 90
```

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
│   ├── core/           # Config, canonical JSON, DAG, checkpoint
│   │   └── landscape/  # Audit trail (the backbone)
│   ├── engine/         # Orchestrator, row processor, executors
│   ├── plugins/        # Sources, transforms, sinks
│   └── cli.py          # Typer CLI
└── tests/
    └── contracts/      # Protocol contract tests
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| CLI | Typer | Commands: run, explain, validate, resume |
| Config | Dynaconf + Pydantic | Multi-source with validation |
| Plugins | pluggy | Extensible pipeline components |
| Audit | SQLAlchemy Core | SQLite (dev) / PostgreSQL (prod) |
| LLM | LiteLLM | 100+ providers unified |

See [Architecture Documentation](ARCHITECTURE.md) for C4 diagrams and detailed design.

---

## Documentation

| Document | Audience | Content |
|----------|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developers | C4 diagrams, data flows, component details |
| [PLUGIN.md](PLUGIN.md) | Plugin Authors | How to create sources, transforms, sinks |
| [REQUIREMENTS.md](REQUIREMENTS.md) | All | System requirements and dependencies |
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
uv pip install -e ".[dev]"

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
