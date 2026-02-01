# CLAUDE.md

## Project Overview

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines**. It provides scaffolding for data processing workflows where every decision must be traceable to its source, regardless of whether the "decide" step is an LLM, ML model, rules engine, or threshold check.

**Current Status:** RC-2. Core architecture, plugins, and audit trail are complete. Stabilization fixes have been integrated and the system is preparing for release.

## Auditability Standard

ELSPETH is built for **high-stakes accountability**. The audit trail must withstand formal inquiry.

**Guiding principles:**

- Every decision must be traceable to source data, configuration, and code version
- Hashes survive payload deletion - integrity is always verifiable
- "I don't know what happened" is never an acceptable answer for any output
- The Landscape audit trail is the source of truth, not logs or metrics
- No inference - if it's not recorded, it didn't happen

**Data storage points** (non-negotiable):

1. **Source entry** - Raw data stored before any processing
2. **Transform boundaries** - Input AND output captured at every transform
3. **External calls** - Full request AND response recorded
4. **Sink output** - Final artifacts with content hashes

This is more storage than minimal, but it means `explain()` queries are simple and complete.

## Data Manifesto: Three-Tier Trust Model

ELSPETH has three fundamentally different trust tiers with distinct handling rules:

### Tier 1: Our Data (Audit Database / Landscape) - FULL TRUST

**Must be 100% pristine at all times.** We wrote it, we own it, we trust it completely.

- Bad data in the audit trail = **crash immediately**
- No coercion, no defaults, no silent recovery
- If we read garbage from our own database, something catastrophic happened (bug in our code, database corruption, tampering)
- Every field must be exactly what we expect - wrong type = crash, NULL where unexpected = crash, invalid enum value = crash

**Why:** The audit trail is the legal record. Silently coercing bad data is evidence tampering. If an auditor asks "why did row 42 get routed here?" and we give a confident wrong answer because we coerced garbage into a valid-looking value, we've committed fraud.

### Tier 2: Pipeline Data (Post-Source) - ELEVATED TRUST ("Probably OK")

**Type-valid but potentially operation-unsafe.** Data that passed source validation.

- Types are trustworthy (source validated and/or coerced them)
- Values might still cause operation failures (division by zero, invalid date formats, etc.)
- Transforms/sinks **expect conformance** - if types are wrong, that's an upstream plugin bug
- **No coercion** at transform/sink level - if a transform receives `"42"` when it expected `int`, that's a bug in the source or upstream transform

**Why:** Plugins have contractual obligations. If a transform's `output_schema` says `int` and it outputs `str`, that's a bug we fix by fixing the plugin, not by coercing downstream.

**Critical nuance:** Type-safe doesn't mean operation-safe:

```python
# Data is type-valid (int), but operation fails
row = {"divisor": 0}  # Passed source validation ✓
result = 100 / row["divisor"]  # 💥 ZeroDivisionError - wrap this!

# Data is type-valid (str), but content is problematic
row = {"date": "not-a-date"}  # Passed as str ✓
parsed = datetime.fromisoformat(row["date"])  # 💥 ValueError - wrap this!
```

### Tier 3: External Data (Source Input) - ZERO TRUST

**Can be literal trash.** We don't control what external systems feed us.

- Malformed CSV rows, NULLs everywhere, wrong types, unexpected JSON structures
- **Validate at the boundary, coerce where possible, record what we got**
- Sources MAY coerce: `"42"` → `42`, `"true"` → `True` (normalizing external data)
- Quarantine rows that can't be coerced/validated
- The audit trail records "row 42 was quarantined because field X was NULL" - that's a valid audit outcome

**Why:** User data is a trust boundary. A CSV with garbage in row 500 shouldn't crash the entire pipeline - we record the problem, quarantine the row, and keep processing the other 10,000 rows.

### The Trust Flow

```text
EXTERNAL DATA              PIPELINE DATA              AUDIT TRAIL
(zero trust)               (elevated trust)           (full trust)
                           "probably ok"

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ External Source │        │ Transform/Sink  │        │ Landscape DB    │
│                 │        │                 │        │                 │
│ • Coerce OK     │───────►│ • No coercion   │───────►│ • Crash on      │
│ • Validate      │ types  │ • Expect types  │ record │   any anomaly   │
│ • Quarantine    │ valid  │ • Wrap ops on   │ what   │ • No coercion   │
│   failures      │        │   row values    │ we     │   ever          │
│                 │        │ • Bug if types  │ saw    │                 │
│                 │        │   are wrong     │        │                 │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                          │
         │                          │
    Source is the              Operations on row
    ONLY place coercion        values need wrapping
    is allowed                 (values can still fail)
```

### External Call Boundaries in Transforms

**CRITICAL:** Trust tiers are about **data flows**, not plugin types. **Any data crossing from an external system is Tier 3**, regardless of which plugin makes the call.

Transforms that make external calls (LLM APIs, HTTP requests, database queries) create **mini Tier 3 boundaries** within their implementation:

```python
def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    # row enters as Tier 2 (pipeline data - trust the schema)

    # External call creates Tier 3 boundary
    try:
        llm_response = self._llm_client.query(prompt)  # EXTERNAL DATA - zero trust
    except Exception as e:
        return TransformResult.error({"reason": "llm_call_failed", "error": str(e)})

    # IMMEDIATELY validate at the boundary - don't let "their data" travel
    try:
        parsed = json.loads(llm_response.content)
    except json.JSONDecodeError:
        return TransformResult.error({"reason": "invalid_json", "raw": llm_response.content[:200]})

    # Validate structure type IMMEDIATELY
    if not isinstance(parsed, dict):
        return TransformResult.error({
            "reason": "invalid_json_type",
            "expected": "object",
            "actual": type(parsed).__name__
        })

    # NOW it's our data (Tier 2) - add to row and continue
    row["llm_classification"] = parsed["category"]  # Safe - validated above
    return TransformResult.success(row)
```

**The rule: Minimize the distance external data travels before you validate it.**

- ✅ **Validate immediately** - right after the external call returns
- ✅ **Coerce once** - normalize types at the boundary
- ✅ **Trust thereafter** - once validated, it's Tier 2 pipeline data
- ❌ **Don't carry raw external data** - passing `llm_response` to helper methods without validation
- ❌ **Don't defer validation** - "I'll check it later when I use it"
- ❌ **Don't validate multiple times** - if it's validated once, trust it

**Common external boundaries in transforms:**

| External Call Type | Tier 3 Boundary | Validation Pattern |
|-------------------|-----------------|-------------------|
| LLM API response | Response content | Wrap JSON parse, validate type is dict, check required fields |
| HTTP API response | Response body | Wrap request, validate status code, parse and validate schema |
| Database query results | Result rows | Validate row structure, handle missing fields, coerce types |
| File reads (in transform) | File contents | Same validation as source plugins |
| Message queue consume | Message payload | Parse format, validate schema, quarantine malformed messages |

**Example from azure_multi_query_llm.py** (the correct pattern):

```python
# Line 227-236: External call (Tier 3 boundary created)
try:
    response = await self._llm_executor.execute_llm_call(...)
except Exception as e:
    return TransformResult.error(...)  # Wrapped immediately

# Line 241-251: IMMEDIATE validation at boundary
try:
    parsed = json.loads(response.content)
except json.JSONDecodeError:
    return TransformResult.error(...)  # Can't parse - reject immediately

# Line 253-263: Structure type validation (defense against non-dict JSON)
if not isinstance(parsed, dict):
    return TransformResult.error({
        "reason": "invalid_json_type",
        "expected": "object",
        "actual": type(parsed).__name__
    })

# Line 266-274: NOW safe to use - it's validated Tier 2 data
output[output_key] = parsed[json_field]  # No defensive .get() needed
```

From this point forward, `parsed` is treated as Tier 2 pipeline data. No more validation. No `.get()` calls. We trust it because we validated it at the boundary.

### Coercion Rules by Plugin Type

| Plugin Type | Coercion Allowed? | Rationale |
|-------------|-------------------|-----------|
| **Source** | ✅ Yes | Normalizes external data at ingestion boundary |
| **Transform (on row)** | ❌ No | Receives validated data; wrong types = upstream bug |
| **Transform (on external call)** | ✅ Yes | External response is Tier 3 - validate/coerce immediately |
| **Sink** | ❌ No | Receives validated data; wrong types = upstream bug |

### Operation Wrapping Rules

| What You're Accessing | Wrap in try/except? | Why |
|----------------------|---------------------|-----|
| `self._config.field` | ❌ No | Our code, our config - crash on bug |
| `self._internal_state` | ❌ No | Our code - crash on bug |
| `landscape.get_row_state(token_id)` | ❌ No | Our data - crash on corruption |
| `row["field"]` arithmetic/parsing | ✅ Yes | Their data values can fail operations |
| `external_api.call(row["id"])` | ✅ Yes | External system, anything can happen |
| `json.loads(external_response)` | ✅ Yes | External data - validate immediately |
| `validated_dict["field"]` | ❌ No | Already validated at boundary - trust it |

**Rule of thumb:**

- **Reading from Landscape tables?** Crash on any anomaly - it's our data.
- **Operating on row field values?** Wrap operations, return error result, quarantine row.
- **Calling external systems?** Wrap call AND validate response immediately at boundary.
- **Using already-validated external data?** Trust it - no defensive `.get()` needed.
- **Accessing internal state?** Let it crash - that's a bug to fix.

## Plugin Ownership: System Code, Not User Code

**CRITICAL DISTINCTION:** All plugins (Sources, Transforms, Gates, Aggregations, Sinks) are **system-owned code**, not user-provided extensions.

### What This Means

```text
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM-OWNED (Full Trust)                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Sources    │  │  Transforms  │  │    Sinks     │           │
│  │  (CSVSource, │  │ (FieldMapper,│  │  (CSVSink,   │           │
│  │   APISource) │  │  LLMTransform)│  │   DBSink)    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    Engine    │  │  Landscape   │  │   Contracts  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ processes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     USER-OWNED (Zero Trust)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      USER DATA                            │   │
│  │   CSV files, API responses, database rows, LLM outputs    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Implications for Error Handling

| Scenario | Correct Response | WRONG Response |
|----------|------------------|----------------|
| Plugin method throws exception | **CRASH** - bug in our code | Catch and log silently |
| Plugin returns wrong type | **CRASH** - bug in our code | Coerce to expected type |
| Plugin missing expected attribute | **CRASH** - interface violation | Use `getattr(x, 'attr', default)` |
| User data has wrong type | Quarantine row, continue | Crash the pipeline |
| User data missing field | Quarantine row, continue | Crash the pipeline |

### Why This Matters for Audit Integrity

A defective plugin that silently produces wrong results is **worse than a crash**:

1. **Crash:** Pipeline stops, operator investigates, bug gets fixed
2. **Silent wrong result:** Data flows through, gets recorded as "correct," auditors see garbage, trust is destroyed

**Example of the problem:**

```python
# WRONG - hides plugin bugs, destroys audit integrity
try:
    result = transform.process(row, ctx)
except Exception:
    result = row  # "just pass through on error"
    logger.warning("Transform failed, using original row")

# RIGHT - plugin bugs crash immediately
result = transform.process(row, ctx)  # Let it crash
```

If `transform.process()` has a bug, we MUST know about it. Silently passing through the original row means the audit trail now contains data that "looks processed" but wasn't - this is evidence tampering.

### NOT a Plugin Marketplace

ELSPETH uses `pluggy` for clean architecture (hooks, extensibility), NOT to accept arbitrary user plugins:

- Plugins are developed, tested, and deployed as part of ELSPETH
- Plugin code is reviewed with the same rigor as engine code
- Plugin bugs are system bugs - they get fixed in the codebase
- Users configure which plugins to use, they don't write their own

## Core Architecture

### The SDA Model

```text
SENSE (Sources) → DECIDE (Transforms/Gates) → ACT (Sinks)
```

- **Source**: Load data (CSV, API, database, message queue) - exactly 1 per run
- **Transform**: Process/classify data - 0+ ordered, includes Gates for routing
- **Sink**: Output results - 1+ named destinations

### Key Subsystems

| Subsystem | Purpose |
| --------- | ------- |
| **Landscape** | Audit backbone - records every operation for complete traceability |
| **Plugin System** | Uses `pluggy` for extensible Sources, Transforms, Sinks |
| **SDA Engine** | RowProcessor, Orchestrator, RetryManager, ArtifactPipeline |
| **Canonical** | Two-phase deterministic JSON canonicalization for hashing |
| **Payload Store** | Separates large blobs from audit tables with retention policies |
| **Configuration** | Dynaconf + Pydantic with multi-source precedence |
| **Config Contracts** | Settings→Runtime protocol enforcement (see below) |

### Settings→Runtime Configuration Pattern

Configuration uses a two-layer pattern to prevent field orphaning:

```text
USER YAML → Settings (Pydantic) → Runtime*Config (dataclass) → Engine Components
             validation            conversion                    runtime behavior
```

**Why two layers?**

1. **Settings classes** (e.g., `RetrySettings`): Pydantic models for YAML validation
2. **Runtime*Config classes** (e.g., `RuntimeRetryConfig`): Frozen dataclasses for engine use

The P2-2026-01-21 bug showed the problem: `exponential_base` was added to `RetrySettings` but never mapped to the engine. Users configured it, Pydantic validated it, but it was silently ignored at runtime.

### The solution: Protocol-based verification

```python
# contracts/config/protocols.py
@runtime_checkable
class RuntimeRetryProtocol(Protocol):
    """What RetryManager EXPECTS from retry config."""
    @property
    def max_attempts(self) -> int: ...
    @property
    def exponential_base(self) -> float: ...  # mypy catches if missing!

# contracts/config/runtime.py
@dataclass(frozen=True, slots=True)
class RuntimeRetryConfig:
    """Implements RuntimeRetryProtocol."""
    max_attempts: int
    exponential_base: float
    # ... other fields

    @classmethod
    def from_settings(cls, settings: "RetrySettings") -> "RuntimeRetryConfig":
        return cls(
            max_attempts=settings.max_attempts,
            exponential_base=settings.exponential_base,  # Explicit mapping!
        )

# engine/retry.py
class RetryManager:
    def __init__(self, config: RuntimeRetryProtocol):  # Accepts protocol
        self._config = config
```

**Enforcement layers:**

1. **mypy (structural typing)**: Verifies `RuntimeRetryConfig` satisfies `RuntimeRetryProtocol`
2. **AST checker**: Verifies `from_settings()` uses all Settings fields (run: `.venv/bin/python -m scripts.check_contracts`)
3. **Alignment tests**: Verifies field mappings are correct and complete

**Key files:**

| File | Purpose |
| ---- | ------- |
| `contracts/config/protocols.py` | Protocol definitions (what engine expects) |
| `contracts/config/runtime.py` | Runtime*Config dataclasses with `from_settings()` |
| `contracts/config/alignment.py` | Field mapping documentation (`FIELD_MAPPINGS`) |
| `contracts/config/defaults.py` | Default values (`POLICY_DEFAULTS`, `INTERNAL_DEFAULTS`) |
| `tests/core/test_config_alignment.py` | Comprehensive alignment verification |

**Adding a new Settings field (checklist):**

1. Add to Settings class in `core/config.py` (Pydantic model)
2. Add to Runtime*Config in `contracts/config/runtime.py` (dataclass field)
3. Map in `from_settings()` method (explicit assignment)
4. If renamed: document in `FIELD_MAPPINGS` in `alignment.py`
5. If internal-only: document in `INTERNAL_DEFAULTS` in `defaults.py`
6. Run `.venv/bin/python -m scripts.check_contracts` and `pytest tests/core/test_config_alignment.py`

### Composite Primary Key Pattern: nodes Table

**CRITICAL:** The `nodes` table has a composite primary key `(node_id, run_id)`. This means the same `node_id` can exist in multiple runs when the same pipeline runs multiple times.

**Queries touching `node_states` must use `node_states.run_id` directly:**

```python
# WRONG - Ambiguous join when node_id is reused across runs
query = (
    select(calls_table)
    .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
    .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)  # BUG!
    .where(nodes_table.c.run_id == run_id)
)

# CORRECT - Use denormalized run_id on node_states
query = (
    select(calls_table)
    .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
    .where(node_states_table.c.run_id == run_id)  # Direct filter, no ambiguous join
)
```

**Why:** The `node_states` table has a denormalized `run_id` column (schema comment: "Added for composite FK"). Use it directly instead of joining through `nodes` table. The join on `node_id` alone would match multiple nodes rows when `node_id` is reused.

**If you MUST access columns from `nodes` table:** Use composite join with BOTH keys:

```python
.join(
    nodes_table,
    (node_states_table.c.node_id == nodes_table.c.node_id) &
    (node_states_table.c.run_id == nodes_table.c.run_id)
)
```

### DAG Execution Model

Pipelines compile to DAGs. Linear pipelines are degenerate DAGs (single `continue` path). Token identity tracks row instances through forks/joins:

- `row_id`: Stable source row identity
- `token_id`: Instance of row in a specific DAG path
- `parent_token_id`: Lineage for forks and joins

### Schema Contracts (DAG Validation)

Transforms can declare field requirements that are validated at DAG construction:

```yaml
# Source guarantees these fields in output
source:
  plugin: csv
  options:
    schema:
      fields: dynamic
      guaranteed_fields: [customer_id, amount]

# Transform requires these fields in input
transforms:
  - plugin: llm_classifier
    options:
      required_input_fields: [customer_id, amount]
```

The DAG validates that upstream `guaranteed_fields` satisfy downstream `required_input_fields`. For template-based transforms, use `elspeth.core.templates.extract_jinja2_fields()` to discover template dependencies during development.

### Transform Subtypes

| Type | Behavior |
| ---- | -------- |
| **Row Transform** | Process one row → emit one row (stateless) |
| **Gate** | Evaluate row → decide destination(s) via `continue`, `route_to_sink`, or `fork_to_paths` |
| **Aggregation** | Collect N rows until trigger → emit result (stateful) |
| **Coalesce** | Merge results from parallel paths |

#### Aggregation Timeout Behavior

Aggregation triggers fire in two ways:

- **Count trigger**: Fires immediately when row count threshold is reached
- **Timeout trigger**: Checked **before** each row is processed

**Known Limitation (True Idle):** Timeout triggers fire when the next row arrives, not during completely idle periods. If no rows arrive, buffered data won't flush until either:

1. A new row arrives (triggering the timeout check)
2. The source completes (triggering end-of-source flush)

For streaming sources that may never end, combine timeout with count triggers, or implement periodic heartbeat rows at the source level.

## Package Management: uv Required

**STRICT REQUIREMENT:** Use `uv` for ALL package management. Never use `pip` directly.

```bash
# Environment setup
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"      # Development with test tools
uv pip install -e ".[llm]"      # With LLM support
uv pip install -e ".[all]"      # Everything

# Adding dependencies
uv pip install <package>        # Install package
uv pip freeze                   # Show installed packages

# Running tests (always use venv python)
.venv/bin/python -m pytest tests/
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
```

## Development Commands

```bash
# Running tests
.venv/bin/python -m pytest tests/                     # All tests
.venv/bin/python -m pytest tests/unit/                # Unit tests only
.venv/bin/python -m pytest tests/integration/         # Integration tests
.venv/bin/python -m pytest -k "test_fork"             # Tests matching pattern
.venv/bin/python -m pytest -x                         # Stop on first failure

# Type checking and linting
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
.venv/bin/python -m ruff check --fix src/             # Auto-fix

# CLI commands
elspeth run --settings pipeline.yaml --execute        # Execute pipeline
elspeth resume <run_id>                               # Resume interrupted run
elspeth validate --settings pipeline.yaml             # Validate config
elspeth plugins list                                  # List available plugins
elspeth purge --run <run_id>                          # Purge payload data

# TUI-based commands (RC-2: limited functionality)
elspeth explain --run <run_id> --row <row_id>         # Lineage explorer (TUI)
```

## Landscape MCP Analysis Server

**For debugging and investigation**, there's an MCP server that provides read-only access to the audit database. This is especially useful for Claude Code sessions investigating pipeline failures.

```bash
# Run the MCP server (auto-discovers databases in current directory)
elspeth-mcp

# Or specify a database explicitly
elspeth-mcp --database sqlite:///./examples/my_pipeline/runs/audit.db
```

The server automatically finds `.db` files, prioritizing `audit.db` in `runs/` directories (pipeline outputs) and sorting by most recently modified.

**Key tools for emergencies:**

- `diagnose()` - First tool when something is broken. Finds failed runs, stuck runs, high error rates
- `get_failure_context(run_id)` - Deep dive on a specific failure
- `explain_token(run_id, token_id)` - Complete lineage for a specific row

**Full documentation:** See `docs/guides/landscape-mcp-analysis.md` for the complete tool reference, common workflows, and database schema guide.

## Technology Stack

### Core Framework

| Component | Technology | Rationale |
| --------- | ---------- | --------- |
| CLI | Typer | Type-safe, auto-generated help |
| TUI | Textual | Interactive terminal UI for `explain`, `status` |
| Configuration | Dynaconf + Pydantic | Multi-source precedence + validation |
| Plugins | pluggy | Battle-tested (pytest uses it) |
| Data | pandas | Standard for tabular data |
| Database | SQLAlchemy Core | Multi-backend without ORM overhead |
| Migrations | Alembic | Schema versioning |
| Retries | tenacity | Industry standard backoff |

### Acceleration Stack (avoid reinventing)

| Component | Technology | Replaces |
| --------- | ---------- | -------- |
| Canonical JSON | `rfc8785` | Hand-rolled serialization (RFC 8785/JCS standard) |
| DAG Validation | NetworkX | Custom graph algorithms (acyclicity, topo sort) |
| Logging | structlog | Ad-hoc logging (structured events) |
| Rate Limiting | pyrate-limiter | Custom leaky buckets |
| Diffing | DeepDiff | Custom comparison (for verify mode) |
| Property Testing | Hypothesis | Manual edge-case hunting |

### Optional Plugin Packs

| Pack | Technology | Use Case |
| ---- | ---------- | -------- |
| LLM | LiteLLM | 100+ LLM providers unified |
| ML | scikit-learn, ONNX | Traditional ML inference |
| Azure | azure-storage-blob | Azure cloud integration |
| Telemetry | OpenTelemetry, ddtrace | Observability platform integration |

## Telemetry (Operational Visibility)

Telemetry provides **real-time operational visibility** alongside the Landscape audit trail.

**Key distinction:**

- **Landscape**: Legal record, complete lineage, persisted forever, source of truth
- **Telemetry**: Operational visibility, real-time streaming, ephemeral, for dashboards/alerting

**No Silent Failures (Critical Principle):**

Any time an object is polled or has an opportunity to emit telemetry, it MUST either:

1. **Send what it has** - emit the telemetry event normally, OR
2. **Explicitly acknowledge "I have nothing"** - log that telemetry was requested but unavailable

If an exception occurs during emission, the acknowledgment must include the failure reason:

- "Telemetry emission failed: [exception details]"
- Never silently swallow events or exceptions

This applies to:

- `telemetry_emit` callbacks in audited clients (HTTP, LLM)
- `TelemetryManager.emit()` calls
- Exporter failures
- Disabled telemetry states (log once at startup that telemetry is disabled)

**Available exporters:** Console (debugging), OTLP (Jaeger/Tempo/Honeycomb), Azure Monitor, Datadog

**Basic configuration:**

```yaml
telemetry:
  enabled: true
  granularity: rows  # lifecycle | rows | full
  backpressure_mode: block  # block (complete) | drop (fast)
  exporters:
    - name: console
      format: pretty
    - name: otlp
      endpoint: ${OTEL_ENDPOINT}
```

**Granularity levels:**

- `lifecycle`: Run start/complete, phase transitions (~10-20 events/run)
- `rows`: Above + row creation, transform completion, gate routing (N x M events)
- `full`: Above + external call details (LLM, HTTP, SQL)

**Correlation workflow:** Telemetry events include `run_id` and `token_id`. When an alert fires in Datadog/Grafana, use the `run_id` to investigate with `elspeth explain` or the Landscape MCP server.

**Full documentation:** See `docs/guides/telemetry.md` for exporter configuration, troubleshooting, and operational guidance.

## Critical Implementation Patterns

### Canonical JSON - Two-Phase with RFC 8785

**NaN and Infinity are strictly rejected, not silently converted.** This is defense-in-depth for audit integrity:

```python
import rfc8785

# Two-phase canonicalization
def canonical_json(obj: Any) -> str:
    normalized = _normalize_for_canonical(obj)  # Phase 1: pandas/numpy → primitives (ours)
    return rfc8785.dumps(normalized)            # Phase 2: RFC 8785/JCS standard serialization
```

- **Phase 1 (our code)**: Normalize pandas/numpy types, reject NaN/Infinity
- **Phase 2 (`rfc8785`)**: Deterministic JSON per RFC 8785 (JSON Canonicalization Scheme)

Test cases must cover: `numpy.int64`, `numpy.float64`, `pandas.Timestamp`, `NaT`, `NaN`, `Infinity`.

### Terminal Row States

Every row reaches exactly one terminal state - no silent drops:

- `COMPLETED` - Reached output sink
- `ROUTED` - Sent to named sink by gate
- `FORKED` - Split to multiple paths (parent token)
- `CONSUMED_IN_BATCH` - Aggregated into batch
- `COALESCED` - Merged in join
- `QUARANTINED` - Failed, stored for investigation
- `FAILED` - Failed, not recoverable

### Retry Semantics

- `(run_id, row_id, transform_seq, attempt)` is unique
- Each attempt recorded separately
- Backoff metadata captured

### Settings→Runtime Field Mapping

**P2-2026-01-21 lesson:** Settings fields can be orphaned (validated but never used at runtime).

```python
# WRONG - Field exists in Settings but not wired to engine
class RetrySettings(BaseModel):
    exponential_base: float = 2.0  # Validated but ignored!

# CORRECT - Explicit from_settings() mapping
@dataclass
class RuntimeRetryConfig:
    exponential_base: float

    @classmethod
    def from_settings(cls, s: RetrySettings) -> "RuntimeRetryConfig":
        return cls(exponential_base=s.exponential_base)  # Explicit!
```

**Verification:** Run `.venv/bin/python -m scripts.check_contracts` and `pytest tests/core/test_config_alignment.py`.

See "Settings→Runtime Configuration Pattern" in Core Architecture for full documentation.

### Secret Handling

Never store secrets - use HMAC fingerprints:

```python
fingerprint = hmac.new(fingerprint_key, secret.encode(), hashlib.sha256).hexdigest()
```

### Test Path Integrity

**Never bypass production code paths in tests.** When integration tests manually construct objects instead of using production factories, bugs hide in the untested path.

**The Dual Code Path Problem:**

```python
# WRONG - Manual construction in tests bypasses production logic
def test_fork_coalesce_manually_built():
    graph = ExecutionGraph()
    graph.add_node("source", ...)
    graph._branch_to_coalesce = {"path_a": "merge1"}  # Manual assignment
    # This test passes even when from_plugin_instances() is broken!
```

```python
# CORRECT - Uses production path
def test_fork_coalesce_production_path():
    graph = ExecutionGraph.from_plugin_instances(  # Production factory
        source=source,
        transforms=transforms,
        sinks=sinks,
        gates=gates,
        coalesce_settings=coalesce_settings,
        output_sink="output",
    )
    branch_map = graph.get_branch_to_coalesce_map()
    # This test FAILS if from_plugin_instances() is broken!
```

**Why this matters:**

- **BUG-LINEAGE-01** hid for weeks because tests manually built graphs
- Manual construction had `branch_to_coalesce[branch] = coalesce_config.name` (correct)
- Production path had `branch_to_coalesce[branch] = cid` (node_id - wrong!)
- Tests passed, production was broken

**Rules:**

- ✅ Use `ExecutionGraph.from_plugin_instances()` in integration tests
- ✅ Use `instantiate_plugins_from_config()` to get real plugin instances
- ✅ Exercise the same code path that production uses
- ❌ Manual `graph.add_node()` / `graph._field = value` bypasses validation
- ❌ Direct attribute assignment skips production logic
- ❌ "It's easier to test this way" creates blind spots

**When manual construction is acceptable:**

- Unit tests of graph algorithms (topological sort, cycle detection)
- Testing graph visualization/rendering
- Testing helper methods that don't depend on construction path

**For integration tests:** Always use production factories.

## Configuration Precedence (High to Low)

1. Runtime overrides (CLI flags, env vars)
2. Suite configuration (`suite.yaml`)
3. Profile configuration (`profiles/production.yaml`)
4. Plugin pack defaults (`packs/llm/defaults.yaml`)
5. System defaults

## The Attributability Test

For any output, the system must prove complete lineage:

```python
lineage = landscape.explain(run_id, token_id=token_id, field=field)
assert lineage.source_row is not None
assert len(lineage.node_states) > 0
```

## Source Layout

```text
src/elspeth/
├── core/
│   ├── landscape/      # Audit trail storage (recorder, exporter, schema)
│   ├── checkpoint/     # Crash recovery checkpoints
│   ├── rate_limit/     # Rate limiting for external calls
│   ├── retention/      # Payload purge policies
│   ├── security/       # Secret fingerprinting via HMAC
│   ├── config.py       # Configuration loading (Dynaconf + Pydantic)
│   ├── canonical.py    # Deterministic JSON hashing (RFC 8785)
│   ├── dag.py          # DAG construction and validation (NetworkX)
│   ├── events.py       # Synchronous event bus for CLI observability
│   └── payload_store.py # Content-addressable storage for large blobs
├── contracts/          # Type contracts, schemas, and protocol definitions
├── engine/
│   ├── orchestrator.py # Full run lifecycle management
│   ├── processor.py    # DAG traversal with work queue
│   ├── executors.py    # Transform, gate, sink, aggregation executors
│   ├── coalesce_executor.py # Fork/join barrier with merge policies
│   ├── artifacts.py    # Artifact pipeline
│   ├── retry.py        # Tenacity-based retry with backoff
│   ├── tokens.py       # Token identity and lineage management
│   ├── triggers.py     # Aggregation trigger evaluation
│   └── expression_parser.py # AST-based expression parsing (no eval)
├── plugins/
│   ├── sources/        # CSVSource, JSONSource, NullSource
│   ├── transforms/     # FieldMapper, Passthrough, Truncate, etc.
│   ├── sinks/          # CSVSink, JSONSink, DatabaseSink
│   ├── llm/            # Azure OpenAI transforms (batch, multi-query)
│   ├── clients/        # HTTP, LLM, Replayer, Verifier clients
│   ├── batching/       # Batch-aware transform adapters
│   └── pooling/        # Thread pool management for plugins
├── tui/                # Terminal UI (Textual) - explain screens and widgets
├── cli.py              # Typer CLI (1700+ LOC)
└── cli_helpers.py      # CLI utility functions
```

## No Legacy Code Policy

**STRICT REQUIREMENT:** Legacy code, backwards compatibility, and compatibility shims are strictly forbidden.

### Anti-Patterns - Never Do This

The following are **strictly prohibited** under all circumstances:

1. **Backwards Compatibility Code**
   - No version checks (e.g., `if version < 2.0: old_code() else: new_code()`)
   - No feature flags for old behavior
   - No "compatibility mode" switches

2. **Legacy Shims and Adapters**
   - No adapter classes to support old interfaces
   - No wrapper functions that translate old APIs to new ones
   - No proxy objects for deprecated functionality

3. **Deprecated Code Retention**
   - No `@deprecated` decorators with code kept around
   - No commented-out old implementations "for reference"
   - No `_legacy` or `_old` suffixed functions

4. **Migration Helpers**
   - No code that supports "both old and new" simultaneously
   - No gradual migration paths in the codebase
   - No transition periods with dual implementations

### The Rule

**When something is removed or changed, DELETE THE OLD CODE COMPLETELY.**

- Don't rename unused variables to `_var` - delete the variable
- Don't keep old code in comments - delete it (git history exists)
- Don't add compatibility layers - change all call sites
- Don't create abstractions to hide breaking changes - make the breaking change

**Default stance:** If old code needs to be removed, delete it completely. If call sites need updating, update them all in the same commit.

### Enforcement

- Claude Code MUST NOT introduce backwards compatibility code
- Claude Code MUST NOT create legacy shims or adapters
- Claude Code MUST delete old code completely when making changes
- Any legacy code patterns MUST be flagged and removed immediately

## Git Safety

**STRICT REQUIREMENT:** Never run destructive git commands without explicit user permission.

### Destructive Commands (REQUIRE PERMISSION)

The following commands can destroy uncommitted work or rewrite history. **ALWAYS ask before running:**

- `git reset --hard` - Discards uncommitted changes
- `git clean -f` - Deletes untracked files permanently
- `git checkout -- <file>` - Discards uncommitted changes to file
- `git stash drop` - Permanently deletes stashed changes
- `git push --force` - Rewrites remote history
- `git rebase` (on pushed branches) - Rewrites shared history

### When You Think You Need a Destructive Command

**Don't.** Go back and get clarification from the user.

## PROHIBITION ON "DEFENSIVE PROGRAMMING" PATTERNS

No Bug-Hiding Patterns: This codebase prohibits defensive patterns that mask bugs instead of fixing them. Do not use .get(), getattr(), hasattr(), isinstance(), or silent exception handling to suppress errors from nonexistent attributes, malformed data, or incorrect types. A common anti-pattern is when an LLM hallucinates a variable or field name, the code fails, and the "fix" is wrapping it in getattr(obj, "hallucinated_field", None) to silence the error—this hides the real bug. When code fails, fix the actual cause: correct the field name, migrate the data source to emit proper types, or fix the broken integration. Typed dataclasses with discriminator fields serve as contracts; access fields directly (obj.field) not defensively (obj.get("field")). If code would fail without a defensive pattern, that failure is a bug to fix, not a symptom to suppress.

### Legitimate Uses

This prohibition does not extend to genuine use cases where defensive handling is necessary:

**1. Operations on Row Values (Their Data)**

Even type-valid row data can cause operation failures. Wrap these operations:

```python
# CORRECT - wrapping operations on their data
def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    try:
        result = row["numerator"] / row["denominator"]  # Their data can be 0
    except ZeroDivisionError:
        return TransformResult.error({"reason": "division_by_zero"})
    return TransformResult.success({"result": result})

# WRONG - wrapping access to OUR internal state
def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    try:
        batch_avg = self._total / self._batch_count  # OUR bug if _batch_count is 0
    except ZeroDivisionError:
        batch_avg = 0  # NO! This hides our initialization bug
```

**The distinction:** Wrapping `row["x"] / row["y"]` is correct because `row` is their data. Wrapping `self._x / self._y` is wrong because `self` is our code.

**2. External System Boundaries**

- **External API responses**: Validating JSON structure from LLM providers or HTTP endpoints before processing
- **Source plugin input**: Coercing/validating external data at ingestion (see Three-Tier Trust Model above)

**3. Framework Boundaries**

- **Plugin schema contracts**: Type checking at plugin boundaries where external code meets the framework
- **Configuration validation**: Pydantic validators rejecting malformed config at load time

**4. Serialization**

- **Pandas dtype normalization**: Converting `numpy.int64` → `int` in canonicalization (already documented above)
- **Serialization polymorphism**: Handling `datetime`, `Decimal`, `bytes` in canonical JSON

### The Decision Test

Ask yourself:

| Question | If Yes | If No |
|----------|--------|-------|
| Is this protecting against user-provided data values? | ✅ Wrap it | — |
| Is this at an external system boundary (API, file, DB)? | ✅ Wrap it | — |
| Would this fail due to a bug in code we control? | — | ❌ Let it crash |
| Am I adding this because "something might be None"? | — | ❌ Fix the root cause |

If you're wrapping to hide a bug that "shouldn't happen," remove the wrapper and fix the bug.

## FINAL COMMENT

If you are thinking to yourself 'we can't break the schema because it will disrupt users' or 'we need to support old data formats', STOP. This codebase has a NO LEGACY CODE POLICY. We do not support backwards compatibility, legacy shims, or compatibility layers. When something is removed or changed, DELETE THE OLD CODE COMPLETELY. Fix all call sites in the same commit. Do not create adapters or compatibility modes. If you need to change a schema, change it fully and update all code that uses it. WE HAVE NO USERS. WE WILL HAVE USERS IN THE FUTURE, DEFERRING BREAKING CHANGES UNTIL WE HAVE USERS IS THE OPPOSITE OF WHAT WE WANT.

If you are proposing or implementing a fix and it involves 'a patch or temporary workaround', STOP. This codebase does not allow patches or temporary workarounds. WE ONLY HAVE ONE CHANCE TO FIX THINGS PRE-RELEASE. Make the fix right, not quick. Do not create 5 hours of technical debt because you wanted to avoid 5 minutes of work today. THIS ESPECIALLY INCLUDES ARCHITECTURAL DEFECTS WHICH MUST BE FIXED PROPERLY NOW. Saying 'I didn't cause this' is not an excuse for disregarding lint, failing tests or CICD. They all must be resolved to merge code, no exceptions. Refusing to do the work now is false economy.
