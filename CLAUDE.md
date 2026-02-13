# CLAUDE.md

## Project Overview

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines**. It provides scaffolding for data processing workflows where every decision must be traceable to its source, regardless of whether the "decide" step is an LLM, ML model, rules engine, or threshold check.

**Current Status:** RC-3. Core architecture, plugins, and audit trail are complete. Quality sprint in progress — stabilization fixes, documentation updates, and contract hardening.

## Auditability Standard

ELSPETH is built for **high-stakes accountability**. The audit trail must withstand formal inquiry.

**Guiding principles:**

- Every decision must be traceable to source data, configuration, and code version
- Hashes survive payload deletion - integrity is always verifiable
- "I don't know what happened" is never an acceptable answer for any output
- The Landscape audit trail is the source of truth, not logs or metrics
- No inference - if it's not recorded, it didn't happen

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
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    # row enters as Tier 2 (pipeline data - trust the schema)

    # External call creates Tier 3 boundary
    try:
        llm_response = self._llm_client.query(prompt)  # EXTERNAL DATA - zero trust
    except Exception as e:
        return TransformResult.error(
            {"reason": "llm_call_failed", "error": str(e)},
            retryable=True,
        )

    # IMMEDIATELY validate at the boundary - don't let "their data" travel
    try:
        parsed = json.loads(llm_response.content)
    except json.JSONDecodeError:
        return TransformResult.error(
            {"reason": "invalid_json", "raw": llm_response.content[:200]},
            retryable=False,
        )

    # Validate structure type IMMEDIATELY
    if not isinstance(parsed, dict):
        return TransformResult.error(
            {
                "reason": "invalid_json_type",
                "expected": "object",
                "actual": type(parsed).__name__
            },
            retryable=False,
        )

    # NOW it's our data (Tier 2) - add to row and continue
    output = {**row.to_dict(), "llm_classification": parsed["category"]}  # Safe - validated above
    return TransformResult.success(
        output,
        success_reason={"action": "llm_classified", "category": parsed["category"]},
    )
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
| `checkpoint_data["tokens"]` | ❌ No | Our data - we wrote this JSON |
| `row["field"]` arithmetic/parsing | ✅ Yes | Their data values can fail operations |
| `external_api.call(row["id"])` | ✅ Yes | External system, anything can happen |
| `json.loads(external_response)` | ✅ Yes | External data - validate immediately |
| `validated_dict["field"]` | ❌ No | Already validated at boundary - trust it |

**Rule of thumb:**

- **Reading from Landscape tables?** Crash on any anomaly - it's our data.
- **Reading checkpoints or deserialized audit JSON?** Crash on any anomaly - it's our data.
- **Operating on row field values?** Wrap operations, return error result, quarantine row.
- **Calling external systems?** Wrap call AND validate response immediately at boundary.
- **Using already-validated external data?** Trust it - no defensive `.get()` needed.
- **Accessing internal state?** Let it crash - that's a bug to fix.

**Serialization does not change trust tier.** Data we wrote to our own database or checkpoint file is still Tier 1 when we read it back, even though it passes through `json.loads()` or SQLAlchemy deserialization. The trust boundary is about *who authored the data*, not the transport format. Checkpoints, audit records, and Landscape tables are all our data — we defined the schema, we wrote the values, we own the invariants. If a deserialized checkpoint is missing a `"tokens"` key or a `"row_id"` field, that is corruption in our system, not a data quality issue to handle gracefully. Crash immediately.

## Plugin Ownership: System Code, Not User Code

All plugins (Sources, Transforms, Aggregations, Sinks) are **system-owned code**, not user-provided extensions. Gates are config-driven system operations, not plugins. ELSPETH uses `pluggy` for clean architecture, NOT to accept arbitrary user plugins. Plugins are developed, tested, and deployed as part of ELSPETH with the same rigor as engine code.

### Implications for Error Handling

| Scenario | Correct Response | WRONG Response |
|----------|------------------|----------------|
| Plugin method throws exception | **CRASH** - bug in our code | Catch and log silently |
| Plugin returns wrong type | **CRASH** - bug in our code | Coerce to expected type |
| Plugin missing expected attribute | **CRASH** - interface violation | Use `getattr(x, 'attr', default)` |
| User data has wrong type | Quarantine row, continue | Crash the pipeline |
| User data missing field | Quarantine row, continue | Crash the pipeline |

A defective plugin that silently produces wrong results is **worse than a crash**:

1. **Crash:** Pipeline stops, operator investigates, bug gets fixed
2. **Silent wrong result:** Data flows through, gets recorded as "correct," auditors see garbage, trust is destroyed

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

### Tier Model Enforcement Allowlist

The allowlist for the tier model enforcement tool (`scripts/cicd/enforce_tier_model.py`) lives in `config/cicd/enforce_tier_model/` as a directory of per-module YAML files:

```text
config/cicd/enforce_tier_model/
├── _defaults.yaml   # version + defaults (fail_on_stale, fail_on_expired)
├── cli.yaml         # per-file rules for cli.py, cli_helpers.py
├── contracts.yaml   # contracts/* entries
├── core.yaml        # core/* entries
├── engine.yaml      # engine/* entries
├── mcp.yaml         # mcp/* entries
├── plugins.yaml     # plugins/* entries
├── telemetry.yaml   # telemetry/* entries
├── testing.yaml     # testing/* entries
└── tui.yaml         # tui/* entries
```

**Adding a new allowlist entry:** Determine the top-level module from the finding's file path (e.g., `core/canonical.py` → `core.yaml`) and add the entry to that module's YAML file under `allow_hits:`.

**The script accepts both a directory and a single file** via `--allowlist`. When no path is given, it prefers the directory if it exists, else falls back to the single-file `enforce_tier_model.yaml`.

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

## Development

**Package management:** Use `uv` for ALL package management. Never use `pip` directly.

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # Development with test tools
uv pip install -e ".[llm]"      # With LLM support
uv pip install -e ".[all]"      # Everything

# Tests and quality
.venv/bin/python -m pytest tests/                     # All tests
.venv/bin/python -m pytest tests/unit/                # Unit tests only
.venv/bin/python -m pytest tests/integration/         # Integration tests
.venv/bin/python -m pytest -k "test_fork"             # Tests matching pattern
.venv/bin/python -m pytest -x                         # Stop on first failure
.venv/bin/python -m mypy src/                         # Type checking
.venv/bin/python -m ruff check src/                   # Linting
.venv/bin/python -m ruff check --fix src/             # Auto-fix lint

# Config contracts verification
.venv/bin/python -m scripts.check_contracts

# Tier model enforcement (defensive pattern detection)
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model

# CLI
elspeth run --settings pipeline.yaml --execute        # Execute pipeline
elspeth resume <run_id>                               # Resume interrupted run
elspeth validate --settings pipeline.yaml             # Validate config
elspeth plugins list                                  # List available plugins
elspeth purge --run <run_id>                          # Purge payload data
elspeth explain --run <run_id> --row <row_id>         # Lineage explorer (TUI)
```

### Landscape MCP Analysis Server

For debugging pipeline failures, an MCP server provides read-only access to the audit database:

```bash
elspeth-mcp                                                    # Auto-discovers databases
elspeth-mcp --database sqlite:///./examples/my_pipeline/runs/audit.db  # Explicit DB
```

**Key tools:** `diagnose()` (what's broken?), `get_failure_context(run_id)` (deep dive), `explain_token(run_id, token_id)` (row lineage). Full reference: `docs/guides/landscape-mcp-analysis.md`.

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

- **Landscape**: Legal record, complete lineage, persisted forever, source of truth
- **Telemetry**: Operational visibility, real-time streaming, ephemeral, for dashboards/alerting

**No Silent Failures:** Any telemetry emission point MUST either send what it has OR explicitly acknowledge "I have nothing" (with failure reason if applicable). Never silently swallow events or exceptions. This applies to `telemetry_emit` callbacks, `TelemetryManager.emit()`, exporter failures, and disabled states (log once at startup).

**Correlation:** Telemetry events include `run_id` and `token_id`. Use these to cross-reference with `elspeth explain` or the Landscape MCP server.

Full configuration guide (exporters, granularity levels, backpressure): `docs/guides/telemetry.md`.

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

### PipelineRow to Dict Conversion

Always use `row.to_dict()` for explicit conversion, not `dict(row)`. Both work (PipelineRow implements the mapping protocol), but `to_dict()` is the established pattern across all transforms.

### Terminal Row States

Every row reaches exactly one terminal state - no silent drops:

- `COMPLETED` - Reached output sink
- `ROUTED` - Sent to named sink by gate
- `FORKED` - Split to multiple paths (parent token)
- `CONSUMED_IN_BATCH` - Aggregated into batch
- `COALESCED` - Merged in join
- `QUARANTINED` - Failed, stored for investigation
- `FAILED` - Failed, not recoverable
- `EXPANDED` - Parent token for deaggregation (1→N expansion)
- `BUFFERED` - Temporarily held in aggregation (non-terminal, becomes COMPLETED on flush)

### Retry Semantics

- `(run_id, row_id, transform_seq, attempt)` is unique
- Each attempt recorded separately
- Backoff metadata captured

### Secret Handling

Never store secrets directly - use HMAC fingerprints for audit:

```python
fingerprint = hmac.new(fingerprint_key, secret.encode(), hashlib.sha256).hexdigest()
```

Secrets can be loaded from environment variables (default) or Azure Key Vault via the `secrets:` section in `settings.yaml`:

```yaml
secrets:
  source: keyvault
  vault_url: https://my-vault.vault.azure.net  # Must be literal URL, not env var reference
  mapping:
    AZURE_OPENAI_KEY: azure-openai-key
    ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key
```

`vault_url` must be a literal HTTPS URL because secrets are loaded before environment variable resolution. Secret resolutions are recorded in the `secret_resolutions` Landscape table (vault source, HMAC fingerprint, latency).

### Test Path Integrity

**Never bypass production code paths in tests.** BUG-LINEAGE-01 hid for weeks because tests manually built `ExecutionGraph` objects instead of using `from_plugin_instances()`. Manual construction had the correct mapping; the production path had a different (wrong) one. Tests passed, production was broken.

```python
# WRONG - bypasses production logic
graph = ExecutionGraph()
graph.add_node("source", ...)
graph._branch_to_coalesce = {"path_a": "merge1"}  # Tests pass, production breaks

# CORRECT - exercises the real code path
graph = ExecutionGraph.from_plugin_instances(source=source, transforms=transforms, ...)
```

**Rules:** Integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()`. Manual construction is acceptable only for unit tests of isolated algorithms (topo sort, cycle detection, visualization).

## Configuration Precedence (High to Low)

1. Runtime overrides (CLI flags, env vars)
2. Pipeline configuration (`settings.yaml`)
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
│   ├── dag/            # DAG construction, validation, graph models (NetworkX)
│   ├── rate_limit/     # Rate limiting for external calls
│   ├── retention/      # Payload purge policies
│   ├── security/       # Secret fingerprinting via HMAC, URL/IP validation
│   ├── config.py       # Configuration loading (Dynaconf + Pydantic)
│   ├── canonical.py    # Deterministic JSON hashing (RFC 8785)
│   ├── events.py       # Synchronous event bus for CLI observability
│   ├── identifiers.py  # ID generation utilities
│   ├── logging.py      # Structured logging setup
│   ├── operations.py   # Operation type definitions
│   ├── payload_store.py # Content-addressable storage for large blobs
│   └── templates.py    # Jinja2 field extraction
├── contracts/          # Type contracts, schemas, and protocol definitions
├── engine/
│   ├── orchestrator/   # Full run lifecycle management (core, aggregation, export, outcomes, validation)
│   ├── executors/      # Transform, gate, sink, aggregation executors
│   ├── processor.py    # DAG traversal with work queue
│   ├── dag_navigator.py # DAG path navigation
│   ├── coalesce_executor.py # Fork/join barrier with merge policies
│   ├── batch_adapter.py # Batch windowing logic
│   ├── retry.py        # Tenacity-based retry with backoff
│   ├── tokens.py       # Token identity and lineage management
│   ├── triggers.py     # Aggregation trigger evaluation
│   ├── expression_parser.py # AST-based expression parsing (no eval)
│   ├── clock.py        # Clock abstraction for testing
│   └── spans.py        # Telemetry span management
├── plugins/
│   ├── sources/        # CSVSource, JSONSource, NullSource, AzureBlobSource
│   ├── transforms/     # FieldMapper, Passthrough, Truncate, etc.
│   ├── sinks/          # CSVSink, JSONSink, DatabaseSink, BlobSink
│   ├── llm/            # Azure OpenAI transforms (batch, multi-query)
│   ├── clients/        # HTTP, LLM, Replayer, Verifier clients
│   ├── batching/       # Batch-aware transform adapters
│   └── pooling/        # Thread pool management for plugins
├── telemetry/          # OpenTelemetry exporters and instrumentation
├── testing/            # ChaosLLM, ChaosWeb, ChaosEngine test servers
├── mcp/                # Landscape MCP analysis server
├── tui/                # Terminal UI (Textual) - explain screens and widgets
├── cli.py              # Typer CLI
├── cli_helpers.py      # CLI utility functions
└── cli_formatters.py   # Event formatting for CLI output
```

## No Legacy Code Policy

**STRICT REQUIREMENT:** Legacy code, backwards compatibility, and compatibility shims are strictly forbidden. WE HAVE NO USERS YET. Deferring breaking changes until we do is the opposite of what we want.

### Anti-Patterns - Never Do This

1. **Backwards Compatibility Code** - No version checks, feature flags for old behavior, or "compatibility mode" switches
2. **Legacy Shims** - No adapter classes, wrapper functions, or proxy objects for deprecated functionality
3. **Deprecated Code Retention** - No `@deprecated` decorators with code kept around, no commented-out implementations "for reference"
4. **Migration Helpers** - No code supporting "both old and new" simultaneously

### The Rule

**When something is removed or changed, DELETE THE OLD CODE COMPLETELY.**

- Don't rename unused variables to `_var` - delete the variable
- Don't keep old code in comments - delete it (git history exists)
- Don't add compatibility layers - change all call sites in the same commit
- Don't create abstractions to hide breaking changes - make the breaking change

If you are proposing a fix that involves "a patch or temporary workaround," STOP. We only have one chance to fix things pre-release. Make the fix right, not quick. This especially includes architectural defects. Lint failures, failing tests, and CI/CD issues must all be resolved to merge — no exceptions.

## Git Safety

**Never run destructive git commands without explicit user permission:**

- `git reset --hard`, `git clean -f`, `git checkout -- <file>` - Discard uncommitted changes
- `git push --force` - Rewrites remote history
- `git rebase` (on pushed branches) - Rewrites shared history

**No git worktrees.** Use regular branches instead.

**No git stash.** The stash/pop cycle has caused repeated data loss in this project — pre-commit hooks that stash/unstash silently destroy unstaged work when `stash pop` encounters conflicts. If you need to preserve work, commit it to a branch.

## Prohibition on Defensive Programming Patterns

This codebase prohibits defensive patterns that mask bugs instead of fixing them. Do not use `.get()`, `getattr()`, `hasattr()`, `isinstance()`, or silent exception handling to suppress errors from nonexistent attributes, malformed data, or incorrect types.

A common anti-pattern: an LLM hallucinates a field name, code fails, and the "fix" is `getattr(obj, "hallucinated_field", None)`. This hides the real bug. Fix the actual cause instead.

**Access typed dataclass fields directly** (`obj.field`), not defensively (`obj.get("field")`). If code would fail without a defensive pattern, that failure is a bug to fix.

### Legitimate Uses

Defensive handling IS appropriate at trust boundaries (see Three-Tier Trust Model above for the full rules and examples):

1. **Operations on row values** - Their data can cause operation failures (division by zero, parse errors). Wrap `row["x"] / row["y"]`, but NOT `self._x / self._y` (our bug if that fails).
2. **External system boundaries** - Validate API/LLM responses immediately at the boundary.
3. **Framework boundaries** - Plugin schema contracts, Pydantic config validation at load time.
4. **Serialization** - pandas/numpy dtype normalization in canonical JSON.

### The Decision Test

| Question | If Yes | If No |
|----------|--------|-------|
| Is this protecting against user-provided data values? | ✅ Wrap it | — |
| Is this at an external system boundary (API, file, DB)? | ✅ Wrap it | — |
| Would this fail due to a bug in code we control? | — | ❌ Let it crash |
| Am I adding this because "something might be None"? | — | ❌ Fix the root cause |

If you're wrapping to hide a bug that "shouldn't happen," remove the wrapper and fix the bug.
