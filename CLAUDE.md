# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines**. It provides scaffolding for data processing workflows where every decision must be traceable to its source, regardless of whether the "decide" step is an LLM, ML model, rules engine, or threshold check.

**Current Status:** Design phase - architecture fully specified, implementation pending.

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

## Development Commands

*Note: Source code implementation pending. Planned commands:*

```bash
# Environment setup
uv venv
source .venv/bin/activate
uv pip install -e ".[llm]"

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

This prohibition does not exclude genuine type handling at system boundaries:

- **External API responses**: Validating JSON structure from LLM providers or HTTP endpoints before processing
- **Plugin schema contracts**: Type checking at plugin boundaries where external code meets the framework
- **Pandas dtype normalization**: Converting `numpy.int64` → `int` in canonicalization (already documented above)
- **Configuration validation**: Pydantic validators rejecting malformed config at load time
- **Serialization polymorphism**: Handling `datetime`, `Decimal`, `bytes` in canonical JSON

**The test**: Ask yourself "is this defensive programming to hide a bug that should not be possible in a well-designed system, or is this legitimate type handling at a trust boundary?" If the former, remove it and fix the underlying issue.
