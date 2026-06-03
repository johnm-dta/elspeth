# ELSPETH

**E**xtensible **L**ayered **S**ecure **P**ipeline **E**ngine for **T**ransformation and **H**andling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
![Status: RC-5.2](https://img.shields.io/badge/status-RC--5.2-yellow.svg)

Elspeth is a high-assurance pipeline substrate for consequential workflows:
systems where the wrong output can cause operational, legal, safety, financial,
or security harm. It supports two authoring surfaces over one runtime assurance
model: operators can hand-edit reviewable, version-controlled YAML, while
knowledge workers can use authenticated Web Composer authoring driven by an LLM
tool loop. Both surfaces target the same primitives, plugin contracts, runtime
assembly, graph-validation contracts, executor, Landscape audit trail, and
run-accounting model. Validation and audit are core product properties, not
after-the-fact diagnostics. In RC-5, composer-authored pipelines converge through
runtime-shaped validation and production execution setup; the longer-term
compiler direction is to seal YAML and composer input into one compiled
artifact that the executor runs directly.

---

## Table of Contents

- [Why Elspeth Exists](#why-elspeth-exists)
- [Architecture At A Glance](#architecture-at-a-glance)
- [What Changed In RC-5](#what-changed-in-rc-5)
  - [RC-5.1 Updates](#rc-51-updates)
  - [RC-5.2 Updates](#rc-52-updates)
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

In RC-5, the important current-state claim is precise:

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

## What Changed In RC-5

RC-5 moves Elspeth from a CLI-first auditable pipeline engine to a dual-surface
authoring and execution platform. The YAML operator path remains first-class,
while the authenticated Web Composer adds LLM-assisted authoring through audited
tools, session state, blobs, secret references, runtime-shaped preview and
preflight validation, background execution, cancellation, diagnostics, WebSocket
progress, and artifact retrieval.

The assurance work also moved forward: runtime-shaped validation, declaration
trust, VAL manifests, terminal outcome modelling, run-accounting invariants,
strict response schemas, and CI policy gates are now part of the product
surface.

### RC-5.1 Updates

RC-5.1 is a correctness and assurance follow-up to RC-5 rather than a new
surface release. The notable deltas:

- **`identity_node_advisory` validator** — `validate_pipeline` now detects
  identity passthroughs wired between transforms and observed sinks, where the
  passthrough silently degrades observed-sink lineage. Surfaces as an
  actionable repair hint in the composer; gated by an exemption matrix locked
  in by tests.
- **Composer pipeline recipes, source inspection, and forced repair** — new
  `apply_pipeline_recipe` MCP tool and template library, an `inspect_source`
  MCP tool that surfaces external-data shape and silent-failure modes as
  warnings, and a forced-repair loop driven by `proof_diagnostics` that fires
  on the resumed-session first turn.
- **Run outputs panel and artifact preview** — the frontend now exposes the
  full audit-evidence manifest for a run, with downloadable artifacts gated by
  a per-artifact `downloadable` flag and backed by a new
  `/artifacts/preview` execution endpoint.
- **Cancellation-requested badge** — runs whose cancellation has been requested
  but not yet drained carry a distinct badge style, separate from the terminal
  `cancelled` state.
- **`data_dir` resolved to absolute path** — `WebSettings` resolves `data_dir`
  to an absolute path at validation time, removing a class of ambiguity where
  relative paths were interpreted against different working directories.
- **GraphView viewport preservation** — composer GraphView preserves the
  operator's pan/zoom across topology changes; iterative edits no longer reset
  the view.
- **Audit-integrity test coverage** — direct unit coverage for four
  ADR-019-family invariants that previously had zero unit tests:
  `sweep_deferred_invariants_or_crash`, `_validate_token_row_ownership`,
  `link_validation_error_to_row`, and `_REQUIRED_COMPOSITE_FOREIGN_KEYS`. Plus
  closure of residual SSRF blocked-IP coverage in `web_scrape`.
- **Composer accessibility (Tier-1 panel-review pass)** — corrections to
  `aria-controls`, `aria-expanded`, `aria-live` scoping, and focus management
  across the composer surface; SecretsPanel form recovery on `createSecret`
  failure; light-theme `--color-status-empty` override.
- **Composer skill correctness** — multi-commit sweep closing fabrication and
  silent-shape-downgrade loopholes, widening the grounding detector, and
  forbidding identity nodes between transforms and observed sinks.
- **Default `on_validation_failure = discard`** — the per-source default
  validation-failure behaviour is now `discard` with documented quarantine
  semantics, replacing the prior implicit fall-through.
- **Unknown-plugin composer error is actionable** —
  `_prevalidate_plugin_options` surfaces unknown plugin ids as structured,
  actionable rejections instead of silent fail-open.

### RC-5.2 Updates

RC-5.2 turns the Web Composer into a more durable, recoverable authoring system:

- **Guided Composer mode** — a structured authoring path for first-time users,
  with deterministic recipe pre-match and a read-only LLM role for guided state.
- **Durable composer progress** — persisted transcript rows, redacted tool-call
  records, composition-state snapshots, and recovery diffs survive interrupted
  or failed turns.
- **Recovery UX** — operators can resume an interrupted authoring session with
  the transcript, redacted tool evidence, and a before/after pipeline-state
  comparison.
- **Completion gestures and catalog polish** — the composer separates save,
  run, execute, and YAML-export actions, while the catalog is now a searchable
  reference surface.
- **CI and documentation hardening** — release reports, docs cleanup, Playwright
  gating, CodeQL, and `elspeth-lints` checks make the release train easier to
  review and repeat.

The RC-5.2 release documentation is intentionally explicit:

- [Executive Summary](docs/release/executive-summary.md) is the current
  public-sector evaluation brief.
- [Composer Guide](docs/release/composer-guide.md) explains the current web
  authoring experience: guided mode, freeform authoring, readiness checks,
  save-for-review, run, export, and recovery.
- [Platform Architecture](docs/release/platform-architecture.md) explains the
  runtime surfaces, trust boundaries, audit-first behaviour, and adopter
  responsibilities.
- [Audit and Lineage Guarantees](docs/release/guarantees.md) is the current
  assurance surface for audit, lineage, execution, data, identity,
  secret-reference handling, sessions, and Composer authoring.
- [Public-Sector Assessment Mapping](docs/release/assessment-mapping.md)
  records the current evidence against government evaluation touchpoints. It is
  an evidence map, not a claim of formal conformance.

See [CHANGELOG.md](CHANGELOG.md) for the full release notes.

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
elspeth resume <run_id>
```

See [Your First Pipeline](docs/guides/your-first-pipeline.md) for a complete walkthrough.

### Web Composer Path

The Web Composer is available through `elspeth web`. It runs a FastAPI backend
and serves the built React frontend from `src/elspeth/web/frontend/dist/`.

Use it when you want to build, inspect, validate, and execute a pipeline
interactively:

- create authenticated user sessions and keep versioned conversation/state
  history
- upload or create blobs, inspect schema hints, and wire blob-backed sources
- store user secrets and wire `$secret{name}` references without exposing raw
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
python - <<'PY'
import os
from pathlib import Path
from elspeth.web.auth.local import LocalAuthProvider

provider = LocalAuthProvider(
    db_path=Path("data/auth.db"),
    secret_key=os.environ["ELSPETH_WEB__SECRET_KEY"],
)
try:
    provider.create_user(
        user_id="demo",
        password="demo12345",
        display_name="Demo User",
        email="demo@example.com",
    )
    print("Created demo user: demo / demo12345")
except ValueError:
    print("Demo user already exists: demo / demo12345")
PY

# 5) Start the web app
elspeth web --host 127.0.0.1 --port 8451
```

Open `http://127.0.0.1:8451` and sign in with the local demo credentials
`demo` / `demo12345`. Do not reuse these credentials outside local development.

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
  `ELSPETH_WEB__REGISTRATION_MODE=open`. For controlled local setups or closed
  registration, create users directly in `data/auth.db` through
  `LocalAuthProvider.create_user(...)` as shown above.
- Session state is stored in `data/sessions.db`; local auth users are stored in
  `data/auth.db`.
- Run audit data defaults to `data/runs/audit.db`; payloads default to
  `data/payloads/`. Override these with `ELSPETH_WEB__LANDSCAPE_URL` and
  `ELSPETH_WEB__PAYLOAD_STORE_PATH` when you need explicit deployment paths.

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
| LLM and safety | Regular `llm` transform with Azure OpenAI/OpenRouter support, multi-query and provider pooling, RAG retrieval, Azure Content Safety, Prompt Shield |
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

The RC-5 line makes several assurance mechanisms product-visible:

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
  exception channels, catch ordering, audit-evidence typing, and (since
  Phase 8) telemetry-backfill cohort-attribution trailers on commits that
  touch code semantically owned by a different phase. Fresh clones should
  install the no-stash staged-file pre-commit dispatcher with
  `scripts/git-hooks/install-pre-commit-dispatcher.sh`, then install the
  commit-msg dispatcher with `scripts/git-hooks/install-commit-msg-dispatcher.sh`.
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

The RC-5 line is where the Web Composer becomes a real product surface, but the
structural change is broader than the web UI. The project now has two authoring
paths over a single high-assurance substrate, richer run evidence, declared
plugin contracts, a stronger terminal outcome model, and more mechanical CI
policy around audit integrity.

Current RC-5.2 behaviour:

- YAML remains a first-class operator path.
- The Web Composer builds through discovery, mutation, blob, secret-reference,
  validation, service-side YAML export, and optional advisor tools.
- Composer validation and web execution use runtime-shaped engine setup rather
  than a standalone UI validator.
- The executor still runs from runtime-assembled settings and graph objects, not
  from a persisted compiled artifact.

Direction after RC-5:

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

Elspeth records complete lineage for every row. The audit database captures:

- Source row with content hash
- Every transform applied (input/output hashes)
- Every gate evaluation with condition result
- Final destination and artifact hash

```bash
# Launch lineage explorer TUI (database path required)
elspeth explain --run <run_id> --row <row_id> --database <path/to/audit.db>
```

For programmatic access, query the Landscape database directly using the `LandscapeRecorder` API.

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

Elspeth can run from a published Docker image. Replace `v0.5.1` with the tag
published for the release you are evaluating.

```bash
IMAGE_TAG=v0.5.1

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
| [docs/architecture/requirements.md](docs/architecture/requirements.md) | All | Verified requirements with implementation status |
| [docs/architecture/adr/](docs/architecture/adr/) | Architects | Architecture Decision Records for routing, declaration-trust, terminal outcomes, and other load-bearing decisions |
| [docs/guides/data-trust-and-error-handling.md](docs/guides/data-trust-and-error-handling.md) | Developers | Trust model, external-boundary handling, quarantine, and plugin error semantics |
| [docs/guides/](docs/guides/) | All | Tutorials, MCP analysis guide, data trust model |
| [docs/release/](docs/release/) | Evaluators | Executive summary, Composer guide, platform architecture, guarantees, assessment mapping, release evidence |
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

Built for systems where decisions must be **traceable, reliable, and defensible**.
