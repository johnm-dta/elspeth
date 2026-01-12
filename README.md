# elspeth-rapid

**Auditable Sense/Decide/Act pipelines for high-reliability systems**

elspeth-rapid is a domain-agnostic framework for building data processing workflows where **every decision must be traceable**. Whether you're evaluating tenders with LLMs, monitoring weather sensors, or processing satellite telemetry, elspeth-rapid provides the scaffolding for reliable, auditable pipelines.

## Why elspeth-rapid?

Modern systems increasingly need to make automated decisions on data streams. When those decisions matter - affecting people, resources, or safety - you need to prove how each decision was made.

elspeth-rapid is designed for **high-level attributability**:

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
cd elspeth-rapid

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

## Core Features

### Full Audit Trail (Landscape)

Every operation is recorded:
- Run configuration (resolved, not just referenced)
- Every transform applied to every row
- Every external call (LLM, API, ML inference)
- Every routing decision with reason
- Every artifact produced

This isn't optional telemetry. It's the core of the system.

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

## Architecture

```
elspeth-rapid/
├── src/elspeth_rapid/
│   ├── core/
│   │   ├── landscape/      # Audit trail storage
│   │   ├── config.py       # Configuration loading
│   │   └── canonical.py    # Deterministic hashing
│   ├── engine/
│   │   ├── runner.py       # SDA pipeline execution
│   │   ├── row_processor.py
│   │   └── artifact_pipeline.py
│   ├── plugins/
│   │   ├── sources/        # Data input plugins
│   │   ├── transforms/     # Processing plugins
│   │   └── sinks/          # Output plugins
│   └── cli.py
├── tests/
└── docs/
    └── design/
        └── architecture.md  # Detailed design document
```

## The Audit Promise

For any output, elspeth-rapid can answer:

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

## When to Use elspeth-rapid

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
