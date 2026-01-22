# ELSPETH Architecture Analysis - Final Report

**Analysis Date:** 2026-01-21
**Analyst:** Claude Code (Opus 4.5)
**Scope:** Full codebase analysis for RC-1 readiness assessment

---

## Executive Summary

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines** designed for high-stakes accountability scenarios where every decision must be traceable to source data, configuration, and code version.

### Key Findings

| Aspect | Assessment |
|--------|------------|
| **Architecture Quality** | HIGH - Clean subsystem boundaries, well-defined contracts, comprehensive audit trail |
| **Code Maturity** | RC-1 READY - Core features complete, LLM integration in Phase 6 of 7 |
| **Documentation** | EXCELLENT - Comprehensive CLAUDE.md, USER_MANUAL, PLUGIN guide, TEST_SYSTEM docs |
| **Test Coverage** | EXTENSIVE - 201 test files, property testing with Hypothesis, contract tests |
| **Technical Debt** | LOW - Explicit "no legacy code" policy enforced |

### Architecture Strengths

1. **Audit-First Design**: Every operation recorded before/after; terminal states derived from audit tables
2. **Three-Tier Trust Model**: Clear enforcement of data trust boundaries at code level
3. **Contract-Driven Development**: 60+ shared types in contracts subsystem prevent integration bugs
4. **Plugin System Clarity**: System-owned plugins (not user extensions) simplifies error handling
5. **Production Features**: Checkpoint/recovery, rate limiting, retention/purge ready for production

### Areas for Attention

1. **Large Files**: `recorder.py` (2,571 LOC), `orchestrator.py` (1,622 LOC), `config.py` (1,186 LOC) could benefit from extraction
2. **Code Duplication**: LLM transforms and Azure pooling contain similar patterns
3. **TUI Incomplete**: Widgets exist but not fully wired into main application
4. **Repository Layer Inconsistent**: Defined but not consistently used in Landscape

---

## Architectural Overview

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        ELSPETH Framework                         │
│                                                                  │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                   │
│  │ SENSE   │────▶│ DECIDE  │────▶│  ACT    │                   │
│  │ Sources │     │ Trans-  │     │ Sinks   │                   │
│  │         │     │ forms/  │     │         │                   │
│  │         │     │ Gates   │     │         │                   │
│  └─────────┘     └─────────┘     └─────────┘                   │
│       │               │               │                          │
│       └───────────────┴───────────────┘                          │
│                       │                                          │
│                       ▼                                          │
│              ┌─────────────────┐                                 │
│              │   LANDSCAPE     │                                 │
│              │  (Audit Trail)  │                                 │
│              │   16 Tables     │                                 │
│              └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Subsystem Summary

| Subsystem | LOC | Purpose | Independence |
|-----------|-----|---------|--------------|
| Contracts | ~2,000 | Shared types crossing subsystem boundaries | HIGH |
| Core Utilities | ~2,000 | Canonical JSON, config, DAG, logging, payload store | MEDIUM |
| Landscape | ~4,600 | Complete audit trail with 16 tables | HIGH |
| Engine | ~5,900 | Pipeline orchestration and token processing | MEDIUM |
| Plugin System | ~2,300 | Protocols, base classes, discovery | HIGH |
| Plugin Implementations | ~7,000 | Sources, transforms, sinks, LLM, Azure | HIGH |
| Production Ops | ~1,200 | Checkpoint, retention, rate limit, security | HIGH |
| CLI/TUI | ~2,000 | User interface | MEDIUM |
| **Total** | **~25,000** | | |

---

## Key Architectural Patterns

### 1. Token-Based DAG Execution

ELSPETH uses a sophisticated token identity system for tracking rows through fork/join operations:

```
row_id       → Stable source row identity (same across forks)
token_id     → Instance of row in specific DAG path (unique per fork)
parent_token_id → Lineage chain for forks and joins
```

This enables:
- Complete lineage tracking through arbitrary DAG topologies
- Proper attribution of processing to specific execution paths
- Support for aggregation (N→1) and deaggregation (1→N) transforms

### 2. Terminal State Recording

Every token reaches exactly one terminal state - no silent drops:

| State | Meaning |
|-------|---------|
| `COMPLETED` | Reached output sink successfully |
| `ROUTED` | Gate sent to named sink |
| `FORKED` | Split to multiple paths (parent terminal) |
| `CONSUMED_IN_BATCH` | Aggregated into batch |
| `COALESCED` | Merged from parallel paths |
| `QUARANTINED` | Failed validation, stored for investigation |
| `FAILED` | Unrecoverable processing error |

### 3. Three-Tier Trust Model

Strictly enforced throughout the codebase:

| Tier | Data | Handling | Example |
|------|------|----------|---------|
| **Tier 1** | Audit DB | CRASH on anomaly | `_row_to_node_state()` raises on NULL output_hash |
| **Tier 2** | Pipeline Data | No coercion, wrap operations | Transforms expect valid types |
| **Tier 3** | External Data | Coerce, validate, quarantine | Sources use `allow_coercion=True` |

### 4. Two-Phase Canonical JSON

Deterministic hashing via RFC 8785:

```python
def canonical_json(obj):
    normalized = _normalize_for_canonical(obj)  # Phase 1: pandas/numpy → primitives
    return rfc8785.dumps(normalized)            # Phase 2: RFC 8785 standard
```

**Key**: NaN and Infinity are **rejected**, not silently converted.

### 5. Plugin Ownership Model

All plugins are **system-owned code**, not user extensions:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM-OWNED (Full Trust)                    │
│  Sources, Transforms, Gates, Sinks, Engine, Landscape           │
├─────────────────────────────────────────────────────────────────┤
│                     USER-OWNED (Zero Trust)                      │
│  CSV files, API responses, database rows, LLM outputs           │
└─────────────────────────────────────────────────────────────────┘
```

Implication: Plugin bugs are system bugs - they crash, not silently degrade.

---

## Technology Stack Assessment

### Core Framework

| Component | Technology | Assessment |
|-----------|------------|------------|
| CLI | Typer | ✓ Type-safe, auto-generated help |
| TUI | Textual | ✓ Modern terminal UI, reactive |
| Configuration | Dynaconf + Pydantic | ✓ Multi-source with validation |
| Plugins | pluggy | ✓ Battle-tested (pytest uses it) |
| Data | pandas | ✓ Standard for tabular data |
| Database | SQLAlchemy Core | ✓ Multi-backend without ORM overhead |
| Migrations | Alembic | ⚠ Present but migrations not visible |
| Retries | tenacity | ✓ Industry standard backoff |

### Acceleration Stack

| Component | Technology | Assessment |
|-----------|------------|------------|
| Canonical JSON | rfc8785 | ✓ Standards-compliant |
| DAG Validation | NetworkX | ✓ Proven graph algorithms |
| Observability | OpenTelemetry | ✓ Industry standard |
| Logging | structlog | ✓ Structured events |
| Rate Limiting | pyrate-limiter | ⚠ Known cleanup race (workaround in place) |
| Diffing | DeepDiff | ✓ For verify mode |
| Property Testing | Hypothesis | ✓ Excellent for invariant testing |

### Optional Packs

| Pack | Technology | Status |
|------|------------|--------|
| LLM | LiteLLM, OpenAI, OpenRouter | Phase 6 (Active) |
| Azure | azure-storage-blob, identity | Phase 7 (Active) |

---

## Data Flow Analysis

### Pipeline Execution Sequence

1. **Initialization**: CLI loads config, validates DAG, instantiates plugins
2. **Run Begin**: Orchestrator creates run record, registers nodes/edges
3. **Source Loading**: Source yields rows; each creates token with audit record
4. **Transform Chain**: Each transform creates node_state with input/output hashes
5. **Gate Routing**: Routing events recorded with edge references
6. **Fork/Join**: Token parents tracked for lineage; coalesce barriers merge
7. **Aggregation**: Engine buffers rows, triggers flush, batch records created
8. **Sink Writes**: Artifacts recorded with content hashes
9. **Completion**: Run status updated, checkpoints deleted on success

### Audit Trail Completeness

The `explain()` function can answer for any token:
- What source row it came from
- What transforms it passed through
- What input/output data at each step
- What external calls were made
- Why it was routed where it was
- What its final outcome was

---

## Risk Assessment

### Low Risk Areas

| Area | Rationale |
|------|-----------|
| Contracts | Pure data models, extensively tested |
| Canonical | RFC 8785 standard, simple implementation |
| Plugin Protocols | Well-defined interfaces, contract tests |
| Core Config | Pydantic validation, environment variable support |

### Medium Risk Areas

| Area | Risk | Mitigation |
|------|------|------------|
| Landscape Recorder | 2,571 LOC complex state | Extensive test coverage visible |
| Orchestrator | Resume logic duplication | Refactoring opportunity identified |
| LLM Integration | External API dependencies | Rate limiting, retry logic in place |
| Azure Batch | Two-phase checkpoint pattern | Documented but complex |

### Areas Requiring Attention

| Area | Issue | Recommendation |
|------|-------|----------------|
| TUI | Widgets not wired to main app | Complete wiring before RC-1 |
| Alembic Migrations | Not visible in analysis | Verify migration strategy |
| Repository Layer | Inconsistent usage | Consolidate or remove |
| Large Files | Maintenance burden | Extract into focused modules |

---

## Compliance and Audit Readiness

### Auditability Standard Met

✓ Every decision traceable to source data
✓ Configuration and code version captured per run
✓ Hashes survive payload deletion
✓ External calls fully recorded (request/response/latency)
✓ Terminal states explicit, not derived

### Export Capabilities

- JSON/CSV export with HMAC signing
- Hash chains for tamper detection
- Grouped or streaming export modes
- Reproducibility grade tracking (FULL/REPLAY/ATTRIBUTABLE)

### Retention Support

- Configurable retention policies
- Purge preserves hashes (audit integrity)
- Grade degradation after purge (ATTRIBUTABLE_ONLY)

---

## Recommendations

### Immediate (Before RC-1)

1. **Complete TUI Wiring**: Wire `LineageTree` and `NodeDetailPanel` into `ExplainApp`
2. **Verify Alembic Setup**: Ensure database migrations are properly configured
3. **Test Resume Path**: Comprehensive testing of checkpoint/recovery

### Short-Term (Post RC-1)

1. **Extract Orchestrator Components**: Split export logic, validation logic into separate modules
2. **Consolidate LLM Patterns**: Create shared base for pooled execution across transforms
3. **Repository Consistency**: Either use repositories consistently or remove the layer

### Long-Term

1. **Schema Versioning**: Consider schema version in Landscape for migration tracking
2. **Distributed Execution**: Current design is single-process; consider distributed patterns
3. **Plugin Marketplace**: If user plugins become a requirement, add sandboxing

---

## Conclusion

ELSPETH demonstrates **excellent architectural quality** with:

- Clear subsystem boundaries and minimal coupling
- Comprehensive audit trail design
- Consistent patterns throughout the codebase
- Strong documentation and testing practices
- Pragmatic technology choices

The codebase is **ready for RC-1** with minor items requiring attention. The "no legacy code" policy has kept technical debt low, and the contract-driven approach prevents many classes of integration bugs.

The architecture is well-suited for its stated purpose: **high-stakes accountability scenarios** where audit integrity is paramount.

---

## Appendices

### A. File Count by Subsystem

```
src/elspeth/contracts/       12 files
src/elspeth/core/            ~25 files (including subdirectories)
src/elspeth/engine/          12 files
src/elspeth/plugins/         ~45 files (including implementations)
src/elspeth/tui/             9 files
src/elspeth/cli.py           1 file
────────────────────────────────────
Total Source:                ~117 files
Total Tests:                 ~201 files
```

### B. External Dependencies (Core)

```
typer>=0.12
textual>=0.52
dynaconf>=3.2
pydantic>=2.6
pluggy>=1.4
pandas>=2.2
sqlalchemy>=2.0
alembic>=1.13
tenacity>=8.2
rfc8785>=0.1
networkx>=3.2
opentelemetry-api>=1.23
structlog>=24.1
pyrate-limiter>=3.1
deepdiff>=7.0
httpx>=0.27
```

### C. Analysis Artifacts

- `00-coordination.md` - Analysis coordination plan
- `01-discovery-findings.md` - Holistic assessment
- `02-subsystem-catalog.md` - Detailed subsystem documentation
- `03-diagrams.md` - Enhanced C4 diagrams
- `04-final-report.md` - This document
- `05-quality-assessment.md` - Code quality evaluation
- `06-architect-handover.md` - Improvement planning document
