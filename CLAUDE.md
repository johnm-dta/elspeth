# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines**. It provides scaffolding for data processing workflows where every decision must be traceable to its source, regardless of whether the "decide" step is an LLM, ML model, rules engine, or threshold check.

**Current Status:** Design phase - architecture fully specified, implementation pending.

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

### Coercion Rules by Plugin Type

| Plugin Type | Coercion Allowed? | Rationale |
|-------------|-------------------|-----------|
| **Source** | ✅ Yes | Normalizes external data at ingestion boundary |
| **Transform** | ❌ No | Receives validated data; wrong types = upstream bug |
| **Sink** | ❌ No | Receives validated data; wrong types = upstream bug |

### Operation Wrapping Rules

| What You're Accessing | Wrap in try/except? | Why |
|----------------------|---------------------|-----|
| `self._config.field` | ❌ No | Our code, our config - crash on bug |
| `self._internal_state` | ❌ No | Our code - crash on bug |
| `row["field"]` arithmetic/parsing | ✅ Yes | Their data values can fail operations |
| `external_api.call(row["id"])` | ✅ Yes | External system, anything can happen |

**Rule of thumb:**
- Reading from Landscape tables? Crash on any anomaly.
- Operating on row field values? Wrap, return error result, quarantine row.
- Accessing internal state? Let it crash - that's a bug to fix.

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

If a future version supports user-authored plugins, those would be sandboxed and treated as untrusted - but that's not the current architecture.

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

### DAG Execution Model

Pipelines compile to DAGs. Linear pipelines are degenerate DAGs (single `continue` path). Token identity tracks row instances through forks/joins:

- `row_id`: Stable source row identity
- `token_id`: Instance of row in a specific DAG path
- `parent_token_id`: Lineage for forks and joins

### Transform Subtypes

| Type | Behavior |
| ---- | -------- |
| **Row Transform** | Process one row → emit one row (stateless) |
| **Gate** | Evaluate row → decide destination(s) via `continue`, `route_to_sink`, or `fork_to_paths` |
| **Aggregation** | Collect N rows until trigger → emit result (stateful) |
| **Coalesce** | Merge results from parallel paths |

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

**Why uv:**

- 10-100x faster than pip
- Deterministic resolution
- Better conflict detection
- Drop-in pip replacement

## Development Commands

```bash
# CLI (planned)
elspeth --settings settings.yaml
elspeth explain --run latest --row 42
elspeth validate --settings settings.yaml
elspeth plugins list
```

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
| Observability | OpenTelemetry + Jaeger | Custom tracing (immediate visualization) |
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

### Secret Handling

Never store secrets - use HMAC fingerprints:

```python
fingerprint = hmac.new(fingerprint_key, secret.encode(), hashlib.sha256).hexdigest()
```

## Configuration Precedence (High to Low)

1. Runtime overrides (CLI flags, env vars)
2. Suite configuration (`suite.yaml`)
3. Profile configuration (`profiles/production.yaml`)
4. Plugin pack defaults (`packs/llm/defaults.yaml`)
5. System defaults

## Implementation Phases

**Design principle:** Prove the DAG infrastructure with deterministic transforms before adding external calls. LLMs are Phase 6, not Phase 1.

| Phase | Priority | Scope |
| ----- | -------- | ----- |
| 1 | P0 | Foundation: Canonical (rfc8785), Landscape, Config, DAG validation (NetworkX) |
| 2 | P0 | Plugin System: hookspecs, base classes, schema contracts |
| 3 | P0 | SDA Engine: RowProcessor, Orchestrator, OpenTelemetry spans |
| 4 | P1 | CLI (Typer + Textual), basic sources/sinks (CSV, JSON, database) |
| 5 | P1 | Production: Checkpointing, rate limiting (pyrate-limiter), retention |
| 6 | P2 | External calls: LLM pack (LiteLLM), record/replay/verify (DeepDiff) |
| 7 | P2 | Advanced: A/B testing, Azure pack, multi-destination routing |

## The Attributability Test

For any output, the system must prove complete lineage:

```python
lineage = landscape.explain(run_id, token_id=token_id, field=field)
assert lineage.source_row is not None
assert len(lineage.node_states) > 0
```

## Planned Source Layout

```text
src/elspeth_rapid/
├── core/
│   ├── landscape/      # Audit trail storage
│   ├── config.py       # Configuration loading
│   └── canonical.py    # Deterministic hashing
├── engine/
│   ├── runner.py       # SDA pipeline execution
│   ├── row_processor.py
│   └── artifact_pipeline.py
├── plugins/
│   ├── sources/        # Data input plugins
│   ├── transforms/     # Processing plugins
│   └── sinks/          # Output plugins
└── cli.py
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

### Rationale

Legacy code and backwards compatibility create:

- **Complexity:** Multiple code paths doing the same thing
- **Confusion:** Unclear which version is "correct"
- **Technical Debt:** Old code that never gets removed
- **Testing Burden:** Must test all combinations
- **Maintenance Cost:** Changes must update both paths

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
