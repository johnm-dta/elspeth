# ELSPETH

**E**xtensible **L**ayered **S**ecure **P**ipeline **E**ngine for **T**ransformation and **H**andling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
![Status: 0.7.2](https://img.shields.io/badge/status-0.7.2-green.svg)

Elspeth is a pipeline engine for building, validating, running, and auditing
workflows where outputs need to be reviewed, explained, and reproduced. It
supports two authoring surfaces over one runtime model: operators can hand-edit
version-controlled YAML, while knowledge workers can use authenticated Web
Composer authoring driven by an LLM tool loop. Both surfaces target the same
primitives, plugin contracts, runtime assembly, graph-validation contracts,
executor, Landscape audit trail, and run-accounting model.

Validation and audit are part of the workflow, not after-the-fact diagnostics.
Composer-authored pipelines use the same validation and execution setup as
YAML-authored pipelines; the longer-term compiler direction is to seal both
inputs into one compiled artifact that the executor runs directly.

**Short walkthrough:** [Watch the Elspeth demo video](docs/video/elspeth.mp4)

The video gives a quick view of what Elspeth is and what it does: the Web
Composer building and validating a pipeline over the same runtime, validation,
and audit model used by YAML-authored pipelines.

---

## Table of Contents

- [Why Elspeth Exists](#why-elspeth-exists)
- [Architecture At A Glance](#architecture-at-a-glance)
- [What Changed In 0.7.2](#what-changed-in-072)
- [Getting Started](#getting-started)
  - [YAML Operator Path](#yaml-operator-path)
  - [Web Composer Path](#web-composer-path)
  - [Frontend Development](#frontend-development)
- [Capabilities](#capabilities)
  - [Authoring And Validation](#authoring-and-validation)
  - [Execution And Run Evidence](#execution-and-run-evidence)
  - [Plugin Surface](#plugin-surface)
  - [MCP Surfaces](#mcp-surfaces)
- [Audit And Assurance](#audit-and-assurance)
  - [Data Trust Model](#data-trust-model)
  - [Audit Trail Export](#audit-trail-export)
  - [JSONL Change Journal](#jsonl-change-journal-optional)
- [Status And Direction](#status-and-direction)
- [Sense/Decide/Act Model](#sensedecideact-model)
- [Example Use Cases](#example-use-cases)
- [Usage](#usage)
  - [Running Pipelines](#running-pipelines)
  - [Explaining Decisions](#explaining-decisions)
- [Landscape MCP Server](#landscape-mcp-server)
- [Configuration](#configuration)
- [Docker](#docker)
- [Repository Architecture](#repository-architecture)
- [Documentation](#documentation)
- [When to Use Elspeth](#when-to-use-elspeth)
- [Contributing](#contributing)
  - [Security And Governance](#security-and-governance)
- [License](#license)

---

## Why Elspeth Exists

High-assurance pipeline tools usually assume the author is already a pipeline
operator: someone comfortable reading YAML, tracing graph edges, and inspecting
runtime evidence. That model fits sensitive, regulated, transactional,
operational, medical, security, or defence-adjacent workflows where every step
must be reviewable and auditable.

LLM workflow builders usually solve a different problem: they make authoring
easier, but often weaken validation, provenance, auditability, and operational
evidence. Elspeth aims to serve both audiences without weakening the assurance
side. The operator path stays hand-edited and version-controlled. The composer
path lets a knowledge worker build document QA, classification, routing,
extraction, reporting, and similar workflows through tools, contracts,
validation, preflight checks, and execution evidence.

The substrate is therefore the product. A pipeline is made from declared
primitives: sources, transforms, pure-config gates, aggregations, coalesce
points, and sinks. Those primitives carry schema and semantic contracts.
Runtime assembly turns them into an execution graph; validation checks wiring,
route targets, schema compatibility, and contracts before the executor runs the
graph and writes the Landscape audit record.

That gives Elspeth two first-class paths:

| Audience | Authoring surface | Why it matters |
| -------- | ----------------- | -------------- |
| Operators in sensitive, regulated, transactional, operational, medical, security, or defence-adjacent workflows | Hand-edited YAML | The pipeline can be read, reviewed, versioned, and explained before it runs |
| Knowledge workers building document QA, classification, routing, extraction, reporting, or review workflows | Authenticated Web Composer | The LLM builds through tools, contracts, validation, preflight checks, and execution evidence rather than emitting unchecked config text |

The Web Composer is therefore not an alternative engine. It is an authoring
surface over the same substrate.

---

## Architecture At A Glance

```text
YAML authoring                         Web Composer authoring
operator-reviewed settings             LLM tool loop + session state
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
       source/transform/sink plugins + pure-config gates
                       │
                       ▼
       plugin schema contracts + route/semantic contracts
                       │
                       ▼
   runtime assembly + graph validation + preflight
                       │
                       ▼
        executor + orchestrator + payload store
                       │
                       ▼
  Landscape audit trail + run accounting + artifacts
```

The current implementation provides these precise guarantees:

- Web validation, web execution, and CLI execution all instantiate runtime
  plugins through the production plugin-instantiation helper and build an
  `ExecutionGraph` through the production graph factory.
- Web validation is not a standalone UI checker. It generates YAML from
  composer state, resolves runtime paths, loads settings, instantiates plugins
  in preflight mode, builds the runtime graph, validates graph structure,
  validates route targets, and validates edge/schema compatibility.
- Composer previews run authoring validation first; when runtime preflight is
  enabled, `preview_pipeline` also calls the same `validate_pipeline()` runtime
  preflight boundary used by the web validation route.
- Web execution reloads persisted composer state, verifies blob ownership and
  runtime-only claims, resolves user secrets in the execution thread, loads
  settings from the generated YAML, builds and validates a runtime graph in
  normal runtime mode, assembles `PipelineConfig`, and runs the orchestrator.
- The CLI path and the web path converge on production plugin instantiation,
  graph construction, graph validation, executor/orchestrator behaviour, and
  Landscape audit evidence. They are not yet one sealed compiler/executor
  artifact boundary.
- The project direction is stronger than the current shape: converge YAML and
  composer input onto a sealed, secret-safe compiled artifact, then have the
  executor run that artifact directly. That compiler boundary is designed, but
  not yet the default execution path.

The assurance machinery is load-bearing in this architecture. Declaration-trust
contracts, runtime VAL manifests, strict response schemas, terminal outcome
modelling, run-accounting invariants, and CI policy gates are what make it
reasonable to let both authoring surfaces feed the same executor.

---

## What Changed In 0.7.2

0.7.2 hardens the production paths introduced in 0.7.1, with emphasis on
bounded trust boundaries and recovery after a worker loses ownership or a
filesystem operation fails after commit.

- **Committed blob deletion is recoverable.** The session store retains the
  exact staged tombstone until unlink and parent-directory fsync complete, so a
  restart can resume direct or failed-fork cleanup without retaining
  unaccounted bytes.
- **Composer evidence remains request-bound.** Provider and prompt identity,
  failed-turn audit evidence, source selection, and fork retention survive
  durable retry without allowing stale or untrusted model data to replace the
  current session state.
- **Multi-worker recovery is more defensive.** Sink-effect preparation and
  fork copying renew leases during I/O, stale workers are fenced after
  ownership loss, and leadership release plus transition responses commit
  atomically.
- **AWS and PostgreSQL defaults are production-aligned.** Bedrock uses the AWS
  default credential chain, both supported PostgreSQL URL driver forms work in
  packaged images, X-Ray identifiers use the compatible format, and acceptance
  manifests bind their final evidence.

**Operational:** 0.7.2 is a pre-1.0 session-store cutover. The session store moves
from epoch 35 to 36; guided schema remains at 10, and Landscape remains at epoch 29.
Archive or export evidence as required, stop the old service, recreate a stale
session store, and install 0.7.2. A Landscape store already at epoch 29 remains
current. Do not roll older code back over the recreated session database.
`data/auth.db` remains separate; recreating the session store does not remove
local user accounts.

See [CHANGELOG.md](CHANGELOG.md) for the complete release-level summary and
[ADR-030](docs/architecture/adr/030-multi-worker-deployment-shape.md) for the
supported multi-worker shape.

---

## Getting Started

Choose the authoring path that matches how you want to work. Both paths end in
an Elspeth pipeline that can be validated, executed, audited, and explained.

### YAML Operator Path

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
elspeth resume <run_id> --execute
```

Without `--execute`, `elspeth resume <run_id>` checks whether the run can be
resumed and reports the resume point without continuing processing.

See [Your First Pipeline](docs/guides/your-first-pipeline.md) for a complete walkthrough.

### Web Composer Path

The Web Composer is available through `elspeth web`. It runs a FastAPI backend
and serves the built React frontend from `src/elspeth/web/frontend/dist/`.

Use it when you want to build, inspect, validate, and execute a pipeline
interactively:

- create authenticated user sessions and keep versioned conversation/state
  history
- upload or create blobs, inspect schema hints, and wire blob-backed sources
- store user secrets and wire `{secret_ref: NAME}` references without exposing raw
  values to the composer
- ask the LLM composer to build or modify a pipeline through audited tools
- preview validation, graph, spec, YAML, semantic contracts, and repair hints
- start a background run, watch progress over WebSocket, cancel safely, and read
  terminal diagnostics and output artifacts

```bash
# 1) Install backend web dependencies
uv pip install -e ".[webui,dev]"

# 2) Build the frontend bundle once
cd src/elspeth/web/frontend
npm install
npm run build
cd ../../../../

# 3) Set a non-default JWT signing key
export ELSPETH_WEB__SECRET_KEY="local-dev-secret-key"

# 4) Create a local demo login user (development only)
elspeth composer users add demo \
  --display-name "Demo User" \
  --email demo@example.com

# 5) Start the web app
elspeth web --host 127.0.0.1 --port 8451
```

When prompted, enter `demo12345` as the demo password. Open
`http://127.0.0.1:8451` and sign in with the local demo credentials `demo` /
`demo12345`. Do not reuse these credentials outside local development.

### Frontend Development

For frontend iteration, run the API server and the Vite dev server separately.
Vite proxies `/api` and `/ws` to the backend on port 8451.

```bash
# Terminal 1
export ELSPETH_WEB__SECRET_KEY="local-dev-secret-key"
elspeth web --host 127.0.0.1 --port 8451

# Terminal 2
cd src/elspeth/web/frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

### Notes

- `elspeth web` requires the `.[webui]` extra. Without it, the CLI exits with
  an install hint.
- The app refuses to start with the default
  `ELSPETH_WEB__SECRET_KEY=change-me-in-production` outside test mode.
- The default auth provider is local auth. Use `--auth oidc` or `--auth entra`
  with the matching `ELSPETH_WEB__*` settings for external identity providers.
- Local auth exposes `/api/auth/register` when
  `ELSPETH_WEB__REGISTRATION_MODE=open` or `email_verified`. The
  `email_verified` mode writes verification links to
  `data/email-verifications.jsonl` for an operator or mailer to deliver.
  For controlled local setups or closed registration, manage users with
  `elspeth composer users add ...` and `elspeth composer users remove ...`.
- Session state is stored in `data/sessions.db`; local auth users are stored in
  `data/auth.db`.
- Run audit data defaults to `data/runs/audit.db`; payloads default to
  `data/payloads/`. Override these with `ELSPETH_WEB__LANDSCAPE_URL` and
  `ELSPETH_WEB__PAYLOAD_STORE_PATH` when you need explicit deployment paths.
- The process-global Prometheus `/metrics` endpoint is disabled by default.
  To enable it, set `ELSPETH_WEB__OPERATOR_METRICS_BEARER_TOKEN` to a dedicated
  operator secret of at least 32 visible ASCII characters (for example,
  `openssl rand -base64 32`) and configure the scraper to send it as a Bearer
  token. Normal user access tokens are never accepted for this endpoint.
- The first-run tutorial scrapes three synthetic pages at
  `{base}/tutorial-site/project-N.html`, where `{base}` defaults to the
  project's public GitHub Pages copy (`https://johnm-dta.github.io/elspeth`).
  That base is operator-controlled content that needs no local hosting, so the
  tutorial runs end-to-end on any deployment — including a pure loopback dev box
  — without the app serving the pages itself. The tutorial's `web_scrape` node
  uses the default `allowed_hosts="public_only"` SSRF policy, so it only fetches
  public origins, exactly like any other web-authored pipeline. Override the
  base with `ELSPETH_WEB__TUTORIAL_SAMPLE_BASE_URL` only if you host your own
  copy of the pages (e.g. a fork).

---

## Capabilities

### Authoring And Validation

- **Two authoring surfaces:** hand-edited YAML and the Web Composer both target
  the same runtime substrate.
- **Composer tool loop:** the LLM can discover plugins, mutate pipeline state,
  manage blobs, reference secrets, preview validation, and request advisor
  hints under explicit trigger categories. YAML export is a service-side
  operation over validated composer state, not a free-form LLM emission.
- **Declarative DAG wiring:** every edge is explicitly named and validated. No
  implicit routing conventions are required for graph interpretation.
- **Contract-aware validation:** schema contracts, semantic contracts, route
  targets, source/sink path policy, secret-reference resolution, plugin
  value-source rules, and runtime preflight failures surface before execution
  where they can be checked.
- **Configuration contract discipline:** user settings flow through validated
  settings models into runtime dataclasses so accepted config fields are not
  silently ignored by the engine.

### Execution And Run Evidence

- **Background web execution:** web runs start asynchronously, stream progress
  over WebSocket, and can be cancelled through a bounded run-state transition.
- **Landscape-derived accounting:** run responses split source rows, emitted
  tokens, terminal tokens, routing/disposition counts, and ledger closure
  integrity instead of collapsing everything into one row count.
- **Terminal statuses:** runs distinguish `completed`,
  `completed_with_failures`, `failed`, `empty`, and `cancelled`.
- **Diagnostics and artifacts:** run endpoints expose diagnostics snapshots,
  discard summaries, output artifact manifests, content hashes, and artifact
  content retrieval.
- **Resilient execution:** checkpointing, retry logic with backoff, plugin-level
  concurrency, rate limiting, payload storage, and retention policies are part
  of the runtime model.

### Plugin Surface

Elspeth discovers source, transform, and sink plugins through pluggy.
Aggregations use batch-aware transform plugins. Gates are pure config: named
expressions and route mappings interpreted by the engine, not plugin classes,
pluggy entries, or dynamically registered components.
Plugin authors can declare schema guarantees, required fields, semantic
requirements, and composer assistance metadata so both humans and the composer
can understand how a component may be wired.

Current plugin families include:

| Family | Examples |
| ------ | -------- |
| Sources and sinks | CSV, JSON, text, null, Azure Blob, Dataverse, database, Chroma, local file outputs |
| Row transforms | Field mapping, type coercion, keyword filtering, truncation, line/json expansion |
| LLM, safety, and document ingestion | Regular `llm` transform with Azure OpenAI/OpenRouter support, multi-query and provider pooling, RAG retrieval, Azure Content Safety, Prompt Shield, Azure Document Intelligence extraction, `blob_fetch`, `blob_csv_expand` |
| Batch analytics | `batch_distribution_profile`, `batch_experiment_compare`, `batch_classifier_metrics`, `batch_paired_preference`, `batch_drift_compare`, `batch_outlier_annotator`, `batch_data_quality_report`, `batch_top_k`, `batch_threshold_summary`, `batch_effect_size` |

The old batch-specific LLM transforms, `azure_batch_llm` and
`openrouter_batch_llm`, were retired. Use the regular `llm` transform with
provider pooling or multi-query for LLM throughput, and use the statistical
batch transforms for local, audit-attributable aggregation and evaluation.

### MCP Surfaces

- **Landscape MCP Server:** `elspeth-mcp` gives read-only tools for diagnosing
  failures, explaining tokens, and inspecting performance from the audit
  database.
- **Composer MCP Server:** `elspeth-composer` exposes the composer tool surface
  outside the web UI for agent-driven pipeline authoring.

---

## Audit And Assurance

Elspeth treats the audit trail as the canonical evidentiary record, not as an
optional log. The Landscape database records source rows, node states, external
calls, payload hashes, route decisions, terminal outcomes, and artifact
provenance. Telemetry and logs are secondary: useful for operations, but not the
source of truth.

The current product-level guarantees are summarised in
[Audit and Lineage Guarantees](docs/release/guarantees.md).

Elspeth makes several assurance mechanisms product-visible:

- **Declaration-trust:** plugin declarations that the graph builder trusts are
  also backed by runtime VAL checks, invariant tests, and CI scanners.
- **Runtime VAL manifests:** runtime validation commitments are recorded so a
  later reader can see which declarations and implementation checks were active
  for a run.
- **Two-axis terminal model:** lifecycle outcome and path/provenance are
  separated so "did this row succeed?" and "how did it get there?" are both
  explicit.
- **Strict response schemas:** web execution responses use strict validation and
  forbid unknown fields so internal drift fails loudly instead of being silently
  coerced.
- **Mechanical policy gates:** CI and pre-commit include checks for tier-model
  boundaries, component types, guard symmetry, contract manifests, composer
  exception channels, catch ordering, and audit-evidence typing.
- **Secret discipline:** runtime secrets are resolved at execution boundaries,
  fingerprinted for audit, and not persisted as raw values in pipeline state.

### Data Trust Model

Elspeth enforces a three-tier trust model that governs how data is handled at
every stage of the pipeline:

| Tier | Data Source | Trust Level | Error Strategy |
| ---- | ----------- | ----------- | -------------- |
| **Tier 1** | Audit database, checkpoints, engine-owned runtime records | Full trust | Crash on any anomaly. Bad audit data means corruption or tampering |
| **Tier 2** | Pipeline data after source validation | Elevated trust | Types are valid; wrap operations on row values where content can still fail |
| **Tier 3** | External input, files, API/LLM responses, source data | Zero trust | Validate at the boundary, coerce where allowed, quarantine malformed rows |

Coercion is only allowed at trust boundaries: sources ingesting external data,
and transforms receiving LLM/API responses. Once data enters the pipeline with
valid types, downstream transforms trust those types. Wrong types downstream are
upstream bugs to fix, not data quality issues to handle gracefully.

This means a CSV with garbage in row 500 should not crash a 10,000-row pipeline
(Tier 3: quarantine the row, keep processing). A corrupted audit record should
crash immediately (Tier 1: silently coercing bad audit data would be evidence
tampering).

See [Data Trust and Error Handling](docs/guides/data-trust-and-error-handling.md)
for the complete model with code examples.

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

Signed exports include every record with an HMAC-SHA256 signature, a manifest
with total count and running hash, and timestamps for chain-of-custody review.

### JSONL Change Journal (Optional)

Enable a redundant JSONL change journal to record committed database writes as
an emergency backup. **Disabled by default.** This is not the canonical audit
record; use it only when you need a text-based, append-only backup stream.

```yaml
landscape:
  dump_to_jsonl: true
  dump_to_jsonl_path: ./runs/audit.journal.jsonl
  # Include request/response payloads for LLM/HTTP calls
  dump_to_jsonl_include_payloads: true
```

---

## Status And Direction

Elspeth is a dual-surface authoring and execution platform: a CLI-first
auditable pipeline engine plus a Web Composer for guided authoring, over one
shared execution and audit core.

Current 0.7.2 behaviour:

- YAML remains a first-class operator path.
- The Web Composer builds through discovery, mutation, blob, secret-reference,
  validation, service-side YAML export, and optional advisor tools.
- Guided pipeline creation is LLM-primary: each stage is built by a language
  model through `/guided/chat`, presented in a conversational builder with a
  live verification panel and gated at the wire stage by an advisor sign-off.
- Composer validation and web execution use runtime-shaped engine setup rather
  than a standalone UI validator.
- The executor still runs from runtime-assembled settings and graph objects, not
  from a persisted compiled artifact.
- A run can be driven by a single process or by a leader plus claim-only
  followers across multiple processes on one host (`elspeth join`), backed by
  one WAL SQLite audit database.

Planned direction (design intentions, not release commitments):

- introduce a first-class compiler facade over the existing graph builder
- preview and seal a secret-safe compiled pipeline artifact
- bind compiled artifacts to Landscape provenance
- run web and CLI execution from the verified artifact instead of reparsing YAML
  as the primary runtime input

See [CHANGELOG.md](CHANGELOG.md) for the release-by-release detail.

---

## Sense/Decide/Act Model

```text
SENSE (Sources)  →  DECIDE (Transforms + Pure-Config Gates)  →  ACT (Sinks)
     │                       │                        │
  Load data            Process & classify       Route to outputs
```

| Stage | What It Does | Examples |
| ----- | ------------ | -------- |
| **Sense** | Load data from built-in or custom sources | CSV, JSON, text/blob files, Azure Blob, Dataverse; custom source plugins for APIs or queues |
| **Decide** | Transform and classify rows | LLM query, ML inference, rules engine, pure-config threshold gate |
| **Act** | Route to appropriate outputs | CSV/JSON files, database rows, Azure Blob, Dataverse, Chroma; custom sinks for alerts or review queues |

**Gates are pure config.** They route rows to different sinks based on named
expressions and route mappings that the engine interprets:

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

## Example Use Cases

| Domain | Sense | Decide | Act |
| ------ | ----- | ------ | --- |
| **Tender Evaluation** | CSV of submissions | LLM classification + safety gates | Results CSV, abuse review queue |
| **Document QA** | PDF/text blobs | LLM extraction, rubric checks, statistical summaries | Annotated outputs, exception queue |
| **Weather Monitoring** | Sensor API feed | Threshold + ML anomaly detection | Routine log, warning, emergency alert |
| **Satellite Operations** | Telemetry stream | Anomaly classifier | Routine log, investigation ticket |
| **Financial Compliance** | Transaction feed | Rules engine + ML fraud detection | Approved, flagged, blocked |
| **Content Moderation** | User submissions | Safety classifier | Published, human review, rejected |

Same framework. Different plugins. Audit evidence recorded with each run.

---

## Usage

### Running Pipelines

```bash
# Validate configuration before running
elspeth validate --settings pipeline.yaml

# Execute a pipeline
elspeth run --settings pipeline.yaml --execute

# Resume an interrupted run (run_id is positional)
elspeth resume abc123 --execute

# List available plugins
elspeth plugins list

# Machine-readable catalog and schema details
elspeth plugins list --format json
elspeth plugins inspect source csv --format json
```

### Explaining Decisions

Elspeth records complete lineage for every row. The audit database captures:

- Source row with content hash
- Every transform applied (input/output hashes)
- Every gate evaluation with condition result
- Final destination and artifact hash

```bash
# Launch lineage explorer TUI (database path required)
elspeth explain --run <run_id> --row <row_id> --database <path/to/audit.db>
```

The TUI renders the recorded graph, including branch labels and repeated DAG
joins, as a selectable tree. Arrow keys move through run, branch, node, token,
and status rows; Enter updates the detail panel; `r` refreshes; `q` exits.

For programmatic access, use `elspeth explain --no-tui` or
`elspeth explain --json`.

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
      mode: observed

transforms:
- name: enrich
  plugin: field_mapper
  input: validated            # Connects to source output
  on_success: enriched        # Named output connection
  options:
    schema:
      mode: observed
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

Elspeth handles messy external headers through source-side normalization and sink-side display restoration.

#### Source Normalization

Normalize messy headers (e.g., `"User ID"`, `"CaSE Study1 !!!! xx!"`) to valid Python identifiers at the source boundary:

```yaml
source:
  plugin: csv
  options:
    path: data/input.csv

    # Headers are always normalized to valid Python identifiers automatically
    # e.g., "User ID" → "user_id"

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
      headers:
        user_id: "User ID"
        amount: "Transaction Amount"

      # Option 2: Auto-restore from source (convenience)
      # headers: original
```

| Option              | Use When                                                          |
| ------------------- | ----------------------------------------------------------------- |
| `headers: {map}`    | You need custom output names or don't want source coupling        |
| `headers: original` | You want to restore exact original headers from normalized source |

Transform-added fields (not in source) use their normalized names when restoring.

### Environment Variables

```bash
# Required for production (secret fingerprinting)
export ELSPETH_FINGERPRINT_KEY="your-stable-key"

# Azure Key Vault (alternative to direct key)
export ELSPETH_KEYVAULT_URL="https://your-vault.vault.azure.net/"
export ELSPETH_KEYVAULT_SECRET_NAME="elspeth-fingerprint-key"
# Optional: pin the exact allowed Key Vault URL(s), comma-separated, as an SSRF
# hardening control; when unset, any https *.vault.azure.net host is accepted
export ELSPETH_KEYVAULT_ALLOWED_VAULT_URLS="https://your-vault.vault.azure.net/"

# LLM API keys
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export OPENROUTER_API_KEY="sk-or-..."

# Signed exports
export ELSPETH_SIGNING_KEY="your-signing-key"

# Audit database encryption (SQLCipher)
export ELSPETH_AUDIT_PASSPHRASE="your-audit-passphrase"
```

Elspeth automatically loads `.env` files. Use `--no-dotenv` to skip in CI/CD.

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

Elspeth fingerprints secrets before storing in the audit trail:

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

Elspeth uses **plugin-level concurrency** rather than orchestrator-level parallelism:

- **Orchestrator**: Single-threaded, sequential token processing (deterministic audit trail)
- **Plugins**: Internally parallelize I/O-bound operations (LLM batching, DB bulk writes)

```yaml
concurrency:
  max_workers: 4  # Available for plugin use (e.g., LLM thread pools)
```

This design ensures audit trail integrity while optimizing performance where it matters. See [ADR-001](docs/architecture/adr/001-plugin-level-concurrency.md) for rationale.

As of 0.6.0, a run can additionally span **multiple cooperating processes on
one host**: one leader (source ingest, barrier evaluation, checkpoints,
finalization, sink I/O) and any number of claim-only followers attached via
`elspeth join <run_id>`, all backed by one WAL SQLite audit database. The
orchestrator's deterministic audit trail is preserved because the leader
remains the single writer of ingest, barrier, and finalization state; followers
only claim and process work items. See
[ADR-030](docs/architecture/adr/030-multi-worker-deployment-shape.md) for the
deployment shape and operator requirements.

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

Elspeth can run from a published Docker image. Replace `v0.7.2` with the tag
published for the release you are deploying; use the exact tag for an older
release line when deploying an earlier version.

```bash
IMAGE_TAG=v0.7.2

# Run a pipeline
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/state:/app/state \
  ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} \
  run --settings /app/config/pipeline.yaml --execute

# Health check
docker run --rm ghcr.io/johnm-dta/elspeth:${IMAGE_TAG} health --json
```

| Mount | Purpose |
| ----- | ------- |
| `/app/config` | Pipeline YAML (read-only) |
| `/app/input` | Source data (read-only) |
| `/app/output` | Sink outputs (read-write) |
| `/app/state` | Audit DB, checkpoints (read-write) |

See [Docker Guide](docs/guides/docker.md) for complete deployment documentation.

---

## Repository Architecture

```text
elspeth/
	├── src/elspeth/
	│   ├── core/               # Config, canonical JSON, rate limiting, retention
	│   │   ├── dag/            # DAG construction, validation, graph models (NetworkX)
	│   │   └── landscape/      # Audit repositories, effect ledgers, export, schema
│   ├── contracts/          # Type contracts, schemas, protocol definitions
│   ├── engine/             # Orchestrator, durable scheduler, DAG and effect coordination
│   │   └── executors/      # Transform, gate, sink, aggregation executors
│   ├── plugins/            # Sources, transforms, sinks, LLM integrations
│   ├── mcp/                # Landscape MCP analysis server
	│   ├── testing/            # ChaosLLM, ChaosWeb, ChaosEngine test servers
	│   ├── web/                # FastAPI app, Composer routes, auth/session storage, frontend
	│   ├── tui/                # Terminal UI (Textual)
	│   └── cli.py              # Typer CLI
	├── docs/                   # Active public documentation
	├── website/                # Standalone static marketing site
	└── tests/
    ├── unit/               # Unit tests
    ├── integration/        # Integration tests
    ├── property/           # Hypothesis property-based tests
    ├── e2e/                # End-to-end pipeline tests
    └── performance/        # Benchmarks, stress, scalability tests
```

| Component | Technology | Purpose |
| --------- | ---------- | ------- |
| CLI | Typer | Commands: run, join, explain, validate, resume, purge |
| TUI | Textual | Interactive graph-backed lineage explorer |
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
| [docs/architecture/requirements.md](docs/architecture/requirements.md) | All | Compatibility pointer to current requirement, contract, and assurance sources |
| [docs/architecture/adr/](docs/architecture/adr/) | Architects | Architecture Decision Records for routing, declaration-trust, terminal outcomes, and other load-bearing decisions |
| [docs/guides/data-trust-and-error-handling.md](docs/guides/data-trust-and-error-handling.md) | Developers | Trust model, external-boundary handling, quarantine, and plugin error semantics |
| [docs/guides/](docs/guides/) | All | Tutorials, MCP analysis guide, data trust model |
| [docs/release/](docs/release/) | Evaluators | Executive summary, Composer guide, platform architecture, guarantees, assessment mapping, release evidence, and archive policy |
| [docs/reference/](docs/reference/) | Developers | Configuration reference |
| [docs/runbooks/](docs/runbooks/) | Operators | Deployment and operations |

---

## When to Use Elspeth

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

# Install the git hook dispatchers (pre-commit + commit-msg policy gates)
scripts/git-hooks/install-pre-commit-dispatcher.sh
scripts/git-hooks/install-commit-msg-dispatcher.sh

# Install Azurite (Azure Blob Storage emulator for integration tests)
npm install

# Run tests
.venv/bin/python -m pytest tests/ -v

# Type checking
.venv/bin/python -m mypy src/

# Linting
.venv/bin/python -m ruff check src/
```

### Security And Governance

- Report suspected vulnerabilities through [SECURITY.md](SECURITY.md). Do not
  disclose exploit details in a public issue before a maintainer confirms a safe
  disclosure path.
- Community behaviour expectations are in
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Support boundaries and response expectations are in [SUPPORT.md](SUPPORT.md).
- Project decision-making, release authority, and continuity risks are
  described in [GOVERNANCE.md](GOVERNANCE.md).

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

Built for workflows where decisions need to be **traceable and reviewable**.
