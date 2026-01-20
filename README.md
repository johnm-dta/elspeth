# ELSPETH

**Auditable Sense/Decide/Act pipelines for high-reliability systems**

ELSPETH is a domain-agnostic framework for building data processing workflows where **every decision must be traceable**. Whether you're evaluating tenders with LLMs, monitoring weather sensors, or processing satellite telemetry, ELSPETH provides the scaffolding for reliable, auditable pipelines.

## Why ELSPETH?

Modern systems increasingly need to make automated decisions on data streams. When those decisions matter - affecting people, resources, or safety - you need to prove how each decision was made.

ELSPETH is designed for **high-level attributability**:

> "This evacuation order came from sensor reading X at time T, which triggered threshold Y in rule Z, with full configuration C"

The framework doesn't care whether your "decide" step is an LLM, a machine learning model, a rules engine, or a simple threshold check. It cares that **every output is traceable to its source**.

## The Sense/Decide/Act Model

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   SENSE     │────▶│   DECIDE    │────▶│    ACT      │
│             │     │             │     │             │
│ Load data   │     │ Transform   │     │ Route to    │
│ from source │     │ and classify│     │ sinks       │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │   ROUTING   │
                    │             │
                    │ Gates route │
                    │ to different│
                    │ action paths│
                    └─────────────┘
```

**Sense**: Load data from any source - CSV files, APIs, databases, message queues.

**Decide**: Process through a chain of transforms. Any transform can be a "gate" that routes rows to different destinations based on classification.

**Act**: Route to appropriate sinks. Different classifications trigger different actions - routine logging, alerts, emergency responses, or human review queues.

## Example Use Cases

| Domain | Sense | Decide | Act |
|--------|-------|--------|-----|
| **Tender Evaluation** | CSV of submissions | LLM classification + safety gates | Results CSV, abuse review queue |
| **Weather Monitoring** | Sensor API feed | Threshold + ML anomaly detection | Routine log, warning, emergency alert |
| **Satellite Operations** | Telemetry stream | Anomaly classifier | Routine log, investigation ticket, intervention |
| **Financial Compliance** | Transaction feed | Rules engine + ML fraud detection | Approved, flagged for review, blocked |
| **Content Moderation** | User submissions | Safety classifier | Published, human review, rejected |

Same framework. Different plugins. Full audit trail.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/elspeth-rapid.git
cd ELSPETH

# Create virtual environment
uv venv
source .venv/bin/activate

# Install with optional LLM support
uv pip install -e ".[llm]"
```

### Your First Pipeline

```yaml
# settings.yaml
datasource:
  plugin: csv_local
  options:
    path: data/submissions.csv

sinks:
  results:
    plugin: csv
    options:
      path: output/results.csv

  flagged:
    plugin: csv
    options:
      path: output/flagged_for_review.csv

row_plugins:
  # Gate: Check for suspicious patterns
  - plugin: pattern_gate
    type: gate
    options:
      patterns: ["ignore previous", "disregard instructions"]
    routes:
      suspicious: flagged
      clean: continue

  # Main processing
  - plugin: llm_query
    options:
      model: gpt-4o-mini
      prompt: "Evaluate this submission: {{ text }}"

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  path: ./runs/audit.db
```

```bash
elspeth --settings settings.yaml
```

### Explain Any Decision

```bash
# What happened to row 42?
elspeth explain --run latest --row 42

# Output:
# Row 42: submission_id=TND-2024-0891
# Source: data/submissions.csv (loaded at 2026-01-12T10:30:00)
#
# Transform 1: pattern_gate
#   Input hash: a3f2c1...
#   Result: EJECTED to 'flagged'
#   Reason: {"pattern": "ignore previous", "confidence": 0.98}
#
# Final destination: flagged_for_review.csv
# Artifact hash: 7b2e4f...
```

## Docker Container

ELSPETH is available as a Docker container for production deployments. The container follows a **CLI-first design** where arguments are passed directly to the `elspeth` CLI.

### Quick Start with Docker

```bash
# Show help
docker run ghcr.io/your-org/elspeth

# Check version
docker run ghcr.io/your-org/elspeth --version

# List available plugins
docker run ghcr.io/your-org/elspeth plugins list
```

### Running Pipelines

Mount your configuration and data directories:

```bash
# Run a pipeline
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/your-org/elspeth:v0.1.0 \
  run --settings /app/config/pipeline.yaml --execute

# Validate configuration
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  ghcr.io/your-org/elspeth:v0.1.0 \
  validate --settings /app/config/pipeline.yaml

# Explain a row
docker run --rm \
  -v $(pwd)/state:/app/state:ro \
  ghcr.io/your-org/elspeth:v0.1.0 \
  explain --run latest --row 42 --no-tui
```

### Standard Volume Mounts

| Host Path | Container Path | Mode | Purpose |
|-----------|----------------|------|---------|
| `./config` | `/app/config` | `ro` | Pipeline YAML, settings |
| `./input` | `/app/input` | `ro` | Source data files (CSV, JSON, etc.) |
| `./output` | `/app/output` | `rw` | Sink output files |
| `./state` | `/app/state` | `rw` | SQLite landscape DB, checkpoints, payloads |
| `./secrets` | `/app/secrets` | `ro` | Sensitive config (optional) |

### Environment Variables

Pass secrets and configuration via environment variables:

```bash
docker run --rm \
  -e DATABASE_URL="sqlite:////app/state/landscape.db" \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -e ELSPETH_FINGERPRINT_KEY="${ELSPETH_FINGERPRINT_KEY}" \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/your-org/elspeth:v0.1.0 \
  run --settings /app/config/pipeline.yaml --execute
```

### Using docker-compose

For easier management, use docker-compose:

```yaml
# docker-compose.yaml
services:
  elspeth:
    image: ghcr.io/your-org/elspeth:${IMAGE_TAG:-latest}
    environment:
      - DATABASE_URL=${DATABASE_URL:-sqlite:////app/state/landscape.db}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
    volumes:
      - ./config:/app/config:ro
      - ./input:/app/input:ro
      - ./output:/app/output
      - ./state:/app/state
    command: ["--help"]
```

```bash
# Run a pipeline
docker compose run --rm elspeth run --settings /app/config/pipeline.yaml --execute

# Validate config
docker compose run --rm elspeth validate --settings /app/config/pipeline.yaml

# Check health
docker compose run --rm elspeth health --verbose
```

### Health Checks

The `health` command verifies system readiness:

```bash
# Basic health check
docker run --rm ghcr.io/your-org/elspeth health

# Verbose output
docker run --rm ghcr.io/your-org/elspeth health --verbose

# JSON output (for automation)
docker run --rm ghcr.io/your-org/elspeth health --json
```

Example output:

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

### Image Tags

| Tag Pattern | Example | Use Case |
|-------------|---------|----------|
| `sha-<commit>` | `sha-abc123f` | CI/CD deployments (immutable) |
| `v<version>` | `v0.1.0` | Release versions |
| `latest` | `latest` | Development (avoid in production) |

### Container Registries

Images are published to:

- **GitHub Container Registry**: `ghcr.io/your-org/elspeth`
- **Azure Container Registry**: `<your-acr>.azurecr.io/elspeth` (if configured)

### Configuration in Containers

Pipeline configurations should use absolute container paths:

```yaml
# config/pipeline.yaml
datasource:
  plugin: csv
  options:
    path: /app/input/data.csv  # References mounted input directory

sinks:
  output:
    plugin: csv
    options:
      path: /app/output/results.csv  # References mounted output directory

landscape:
  url: ${DATABASE_URL:-sqlite:////app/state/landscape.db}

payload_store:
  base_path: /app/state/payloads
```

### Building the Image Locally

```bash
# Build the image
docker build -t elspeth:local .

# Run locally built image
docker run --rm elspeth:local --version
```

## Core Features

### Full Audit Trail (Landscape)

Every operation is recorded:

- Run configuration (resolved, not just referenced)
- Every transform applied to every row
- Every external call (LLM, API, ML inference)
- Every routing decision with reason
- Every artifact produced

This isn't optional telemetry. It's the core of the system.

### Audit Trail Export

ELSPETH can automatically export the complete audit trail after each run for compliance and legal inquiry.

#### Configuration

```yaml
landscape:
  url: sqlite:///./runs/audit.db
  export:
    enabled: true
    sink: audit_archive     # Must reference a defined sink
    format: json            # json (note: csv requires homogeneous records)
    sign: true              # HMAC signature per record

sinks:
  audit_archive:
    plugin: json
    options:
      path: exports/audit_trail.json
```

#### Signed Exports

For legal-grade integrity verification, enable signing:

```bash
export ELSPETH_SIGNING_KEY="your-secret-key"
uv run elspeth run -s settings.yaml --execute
```

Each record receives an HMAC-SHA256 signature. A manifest record at the end contains:
- Total record count
- Running hash of all signatures
- Export timestamp

This allows auditors to:
1. Verify no records were added, removed, or modified
2. Trace any row through every processing step
3. Prove chain-of-custody for legal proceedings

#### Export Record Types

The export includes all audit data:

| Record Type | Description |
|-------------|-------------|
| `run` | Run metadata (config hash, timestamps, status) |
| `node` | Registered plugins (source, transforms, sinks) |
| `edge` | Graph edges between nodes |
| `row` | Source rows with content hashes |
| `token` | Row instances in pipeline paths |
| `token_parent` | Fork/join lineage |
| `node_state` | Processing records (input/output hashes) |
| `routing_event` | Gate routing decisions |
| `call` | External API calls |
| `batch` | Aggregation batches |
| `batch_member` | Batch membership |
| `artifact` | Sink outputs |
| `manifest` | Final hash and metadata (signed exports only) |

### Conditional Routing (Gates)

Transforms can route rows to different sinks:

```yaml
row_plugins:
  - plugin: safety_classifier
    type: gate
    routes:
      safe: continue
      prompt_injection: abuse_review
      pii_detected: pii_violations
```

Ejected rows carry their classification metadata to the destination sink.

### Plugin System

Everything is pluggable:

- **Sources**: CSV, database, HTTP API, blob storage
- **Transforms**: LLM query, ML inference, rules check, threshold gate
- **Sinks**: CSV, Excel, database, webhook, archive bundle

Add new capabilities without modifying core code.

### Production Ready

- **Checkpointing**: Resume interrupted runs
- **Retry logic**: Configurable backoff for transient failures
- **Rate limiting**: Respect API limits
- **Concurrent processing**: Multi-threaded row processing
- **A/B testing**: Compare baseline vs variant approaches

## Configuration

### Hierarchical Settings

Configurations merge with clear precedence:

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
      backend: postgresql
```

```bash
elspeth --settings config.yaml --profile production
```

### Environment Variables

```yaml
llm:
  options:
    api_key: ${OPENAI_API_KEY}  # Loaded from environment
```

### Secret Fingerprinting

ELSPETH fingerprints sensitive configuration values (API keys, tokens, passwords) before storing them in the audit trail. This ensures secrets are never written to the database in plain text.

#### Required Environment Variable

```bash
# Production: Set a stable secret key for fingerprinting
export ELSPETH_FINGERPRINT_KEY="your-secret-key-here"
```

**IMPORTANT:** If `ELSPETH_FINGERPRINT_KEY` is not set and your configuration contains secrets, ELSPETH will raise a `SecretFingerprintError` at startup. This is intentional - silent secret leakage to the audit database is a security risk.

#### Development Mode

For local development where fingerprint stability isn't required:

```bash
# Development only: Allow secrets without fingerprinting
export ELSPETH_ALLOW_RAW_SECRETS=true
```

This will redact secrets (replacing them with `[REDACTED]`) instead of fingerprinting them. **Do not use in production.**

#### What Gets Fingerprinted

- Plugin options with secret-like field names (`api_key`, `token`, `password`, `secret`, etc.)
- Nested secrets in configuration objects
- Database passwords in `landscape.url` DSN strings

#### Behavior Change Notice

Prior versions silently preserved raw secrets when `ELSPETH_FINGERPRINT_KEY` was unset. Current versions fail-closed by default - you must either:
1. Set `ELSPETH_FINGERPRINT_KEY` (recommended), or
2. Explicitly opt-in to dev mode with `ELSPETH_ALLOW_RAW_SECRETS=true`

### Automatic .env Loading

ELSPETH automatically loads environment variables from a `.env` file at startup. This means you can store your API keys and secrets in a `.env` file without manually sourcing it before each command.

#### Example .env File

```bash
# .env - Environment variables for ELSPETH
# This file is automatically loaded by the CLI

# API Keys for LLM plugins
OPENROUTER_API_KEY=sk-or-v1-your-key-here
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Secret handling
ELSPETH_FINGERPRINT_KEY=your-stable-fingerprint-key

# Development mode (not for production!)
# ELSPETH_ALLOW_RAW_SECRETS=true

# Signed exports
ELSPETH_SIGNING_KEY=your-signing-key
```

#### How It Works

- The CLI automatically loads `.env` from the current directory (or parent directories)
- Existing environment variables are **not** overwritten (explicit env vars take precedence)
- Uses `python-dotenv` under the hood

#### Skipping .env Loading

In CI/CD environments where secrets are injected externally, you may want to skip `.env` loading:

```bash
# Skip .env loading (useful in CI/CD)
elspeth --no-dotenv run -s settings.yaml --execute
```

#### Security Note

**Never commit `.env` files to version control.** Add `.env` to your `.gitignore`:

```gitignore
# Secrets
.env
.env.local
.env.*.local
```

## Architecture

```
elspeth-rapid/
├── src/elspeth/
│   ├── core/
│   │   ├── landscape/      # Audit trail storage and export
│   │   ├── checkpoint/     # Run checkpointing for recovery
│   │   ├── retention/      # Data retention policies
│   │   ├── security/       # Secret fingerprinting
│   │   ├── rate_limit/     # API rate limiting
│   │   ├── config.py       # Configuration loading
│   │   ├── canonical.py    # Deterministic JSON hashing
│   │   └── dag.py          # Execution graph
│   ├── engine/
│   │   ├── orchestrator.py # Pipeline orchestration
│   │   ├── processor.py    # Row processing loop
│   │   ├── executors.py    # Transform/Gate/Aggregation executors
│   │   ├── coalesce_executor.py  # Fork/join merge logic
│   │   └── triggers.py     # Aggregation trigger evaluation
│   ├── plugins/
│   │   ├── sources/        # Data input plugins (CSV, JSON)
│   │   ├── transforms/     # Processing plugins
│   │   ├── sinks/          # Output plugins (CSV, JSON, DB)
│   │   └── protocols.py    # Plugin interface contracts
│   ├── contracts/          # Type contracts, results, enums
│   ├── tui/                # Terminal UI for explain/status
│   └── cli.py              # Command-line interface
├── tests/
└── docs/
    ├── contracts/
    │   └── plugin-protocol.md  # Plugin interface specification
    ├── runbooks/               # Operational guides
    └── design/
        └── architecture.md     # System design document
```

## The Audit Promise

For any output, ELSPETH can answer:

1. **What was the input?** - Source data with hash
2. **What transforms were applied?** - Full chain with configs
3. **What external calls were made?** - LLM prompts, API calls, ML inferences
4. **Why was this routing decision made?** - Gate evaluation with reason
5. **When did this happen?** - Timestamps throughout
6. **Can we replay it?** - Full config stored, responses recorded, hashes verified

Reproducible to the extent possible - deterministic transforms replay exactly; non-deterministic external calls (LLMs, APIs) can be replayed from recorded responses or re-executed and compared.

Complete chain of custody from input to output.

## Technology Stack

| Component | Technology | Why |
|-----------|------------|-----|
| CLI | Typer | Type-safe, great UX |
| Config | Dynaconf + Pydantic | Multi-source + validation |
| Plugins | pluggy | Battle-tested (pytest uses it) |
| Audit Storage | SQLAlchemy | SQLite dev, PostgreSQL prod |
| LLM (optional) | LiteLLM | 100+ providers unified |

## Documentation

- **[Architecture](docs/design/architecture.md)** - Detailed design document

## When to Use ELSPETH

**Good fit:**

- Decisions that need to be explainable
- Regulatory or compliance requirements
- Systems where "why did it do that?" matters
- Workflows mixing automated and human review

**Consider alternatives if:**

- Pure high-throughput ETL (use Spark, dbt)
- Real-time streaming with sub-second latency (use Flink, Kafka Streams)
- Simple scripts with no audit requirements

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

Built for systems where decisions must be **traceable, reliable, and defensible**.
