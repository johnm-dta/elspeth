# Discovery Findings

**Analysis Date:** 2026-01-21
**Analyst:** Claude Code (Opus 4.5)
**Scope:** Full ELSPETH codebase

## Executive Summary

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines**. Every decision flows through a complete audit trail ensuring traceability to source data, configuration, and code version. The architecture is approaching RC-1 status with LLM integration in Phase 6 of 7.

**Key Finding:** The codebase is well-structured with clear subsystem boundaries, comprehensive contracts, and a sophisticated three-tier trust model. The audit trail (Landscape) is the architectural centerpiece.

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Source Files | 117 (.py in src/) |
| Total Test Files | 201 (.py in tests/) |
| Source LOC | ~25,000 |
| Test LOC | ~30,000 (estimated) |
| Python Version | 3.12+ |
| Major Subsystems | 8 |

## Directory Structure

```
elspeth-rapid/
├── src/elspeth/           # Main source (117 files, ~25K LOC)
│   ├── cli.py             # Typer CLI entry point (1,139 LOC)
│   ├── contracts/         # Pydantic data models (~2K LOC)
│   ├── core/              # Infrastructure (~8K LOC)
│   │   ├── canonical.py   # RFC 8785 JSON canonicalization
│   │   ├── config.py      # Dynaconf + Pydantic config
│   │   ├── dag.py         # DAG validation (NetworkX)
│   │   ├── landscape/     # Audit trail (~3.4K LOC)
│   │   ├── checkpoint/    # Pipeline checkpointing
│   │   ├── rate_limit/    # Rate limiting
│   │   ├── retention/     # Data retention/purge
│   │   └── security/      # Secret fingerprinting
│   ├── engine/            # SDA engine (~5.9K LOC)
│   │   ├── orchestrator.py # Run lifecycle (1,622 LOC)
│   │   ├── processor.py   # Row processing (1,014 LOC)
│   │   ├── executors.py   # Plugin execution (1,340 LOC)
│   │   └── ...
│   ├── plugins/           # Plugin system (~11.7K LOC)
│   │   ├── protocols.py   # Plugin contracts
│   │   ├── base.py        # Base classes
│   │   ├── sources/       # Data sources (CSV, JSON)
│   │   ├── transforms/    # Processing transforms
│   │   ├── sinks/         # Output sinks
│   │   ├── llm/           # LLM integration pack
│   │   ├── azure/         # Azure integration pack
│   │   └── clients/       # HTTP/LLM clients
│   └── tui/               # Textual TUI (~875 LOC)
├── tests/                 # Test suite (201 files)
│   ├── property/          # Hypothesis property tests
│   ├── integration/       # Integration tests
│   └── system/            # System/E2E tests
├── examples/              # Working examples (12 directories)
├── alembic/               # Database migrations
├── config/                # Configuration files
└── docs/                  # Documentation
```

## Technology Stack

### Core Framework

| Component | Technology | Purpose |
|-----------|------------|---------|
| CLI | Typer | Type-safe command-line interface |
| TUI | Textual | Interactive terminal UI for lineage |
| Configuration | Dynaconf + Pydantic | Multi-source config with validation |
| Plugins | pluggy | Battle-tested plugin system |
| Data | pandas | Tabular data manipulation |
| Database | SQLAlchemy Core | Multi-backend without ORM overhead |
| Migrations | Alembic | Schema versioning |
| Retries | tenacity | Industry-standard backoff |

### Acceleration Stack

| Component | Technology | Replaces |
|-----------|------------|----------|
| Canonical JSON | rfc8785 | Hand-rolled serialization |
| DAG Validation | NetworkX | Custom graph algorithms |
| Observability | OpenTelemetry | Custom tracing |
| Logging | structlog | Ad-hoc logging |
| Rate Limiting | pyrate-limiter | Custom leaky buckets |
| Diffing | DeepDiff | Custom comparison |
| Property Testing | Hypothesis | Manual edge-case hunting |

### Optional Plugin Packs

| Pack | Technology | Status |
|------|------------|--------|
| LLM | LiteLLM + OpenAI + OpenRouter | Phase 6 (Active) |
| Azure | azure-storage-blob + identity | Phase 7 (Active) |

## Subsystem Identification

I have identified **8 major subsystems** with clear boundaries:

### 1. Contracts (~2K LOC)
**Location:** `src/elspeth/contracts/`
**Purpose:** Shared data types crossing subsystem boundaries
**Key Files:**
- `audit.py` (419 LOC) - Audit trail data models
- `results.py` (324 LOC) - Plugin result types
- `enums.py` (219 LOC) - Status/type enumerations
- `data.py` (243 LOC) - Schema contracts
- `config.py` (38 LOC) - Configuration models

### 2. Core/Canonical (~1.4K LOC)
**Location:** `src/elspeth/core/canonical.py`, `config.py`, `dag.py`, `logging.py`
**Purpose:** Foundational utilities
**Key Components:**
- Two-phase RFC 8785 JSON canonicalization
- Dynaconf-based configuration with precedence
- DAG validation using NetworkX
- structlog-based logging

### 3. Landscape (Audit Trail) (~3.4K LOC)
**Location:** `src/elspeth/core/landscape/`
**Purpose:** Complete audit trail storage and querying
**Key Files:**
- `recorder.py` (2,567 LOC) - High-level audit API (THE centerpiece)
- `schema.py` (359 LOC) - SQLAlchemy table definitions
- `exporter.py` (382 LOC) - JSON/CSV export
- `lineage.py` (149 LOC) - explain() implementation
- `models.py` (334 LOC) - Internal model helpers

### 4. Engine (~5.9K LOC)
**Location:** `src/elspeth/engine/`
**Purpose:** Pipeline execution runtime
**Key Files:**
- `orchestrator.py` (1,622 LOC) - Run lifecycle management
- `executors.py` (1,340 LOC) - Plugin execution
- `processor.py` (1,014 LOC) - Row-by-row processing
- `expression_parser.py` (464 LOC) - Gate condition parsing
- `coalesce_executor.py` (453 LOC) - Fork/join handling

### 5. Plugin System (~2K LOC)
**Location:** `src/elspeth/plugins/` (base files only)
**Purpose:** Plugin infrastructure
**Key Files:**
- `protocols.py` (456 LOC) - Plugin interface definitions
- `context.py` (375 LOC) - Runtime context
- `base.py` (329 LOC) - Base classes
- `manager.py` (242 LOC) - Discovery/registration

### 6. Plugin Implementations (~7K LOC)
**Location:** `src/elspeth/plugins/sources/`, `transforms/`, `sinks/`, `llm/`, `azure/`
**Purpose:** Concrete plugin implementations
**Notable:**
- Azure Batch LLM (722 LOC) - Batch processing for Azure OpenAI
- OpenRouter LLM (668 LOC) - Multi-model routing
- Content Safety (617 LOC) - Azure AI Content Safety
- Prompt Shield (555 LOC) - Jailbreak detection
- Azure Blob Source/Sink (482/447 LOC) - Azure Blob Storage

### 7. Production Ops (~1.2K LOC)
**Location:** `src/elspeth/core/checkpoint/`, `retention/`, `rate_limit/`, `security/`
**Purpose:** Production reliability features
**Key Components:**
- Checkpoint/Resume with recovery
- Retention/Purge policies
- Rate limiting registry
- Secret fingerprinting (HMAC-based)

### 8. CLI/TUI (~2K LOC)
**Location:** `src/elspeth/cli.py`, `src/elspeth/tui/`
**Purpose:** User interface
**Key Components:**
- Typer CLI with run, explain, validate, purge, resume commands
- Textual TUI for interactive lineage exploration

## Key Architectural Patterns

### 1. Three-Tier Trust Model

```
Tier 1: Our Data (Audit DB)     → CRASH on any anomaly
Tier 2: Pipeline Data           → No coercion, wrap operations
Tier 3: External Data (Sources) → Coerce, validate, quarantine
```

This is **strictly enforced** - the CLAUDE.md makes clear that defensive programming to hide bugs is prohibited.

### 2. Token-Based DAG Execution

```
row_id   → Stable source row identity
token_id → Instance in specific DAG path
parent_token_id → Lineage for forks/joins
```

### 3. Terminal Row States

Every row reaches exactly one terminal state:
- `COMPLETED` - Reached output sink
- `ROUTED` - Sent to named sink by gate
- `FORKED` - Split to multiple paths
- `CONSUMED_IN_BATCH` - Aggregated
- `COALESCED` - Merged in join
- `QUARANTINED` - Failed validation
- `FAILED` - Unrecoverable error

### 4. No Legacy Code Policy

The codebase explicitly forbids:
- Backwards compatibility code
- Legacy shims/adapters
- Deprecated code retention
- Migration helpers

**When something is removed, DELETE COMPLETELY.**

### 5. Plugin Ownership

All plugins are **system-owned code**, not user-provided extensions. Plugin bugs are system bugs.

## Entry Points

| Entry Point | File | Purpose |
|-------------|------|---------|
| CLI | `cli.py:app` | Main Typer application |
| Engine | `engine/orchestrator.py:Orchestrator.run()` | Pipeline execution |
| Audit | `core/landscape/recorder.py:LandscapeRecorder` | Audit API |
| Config | `core/config.py:load_settings()` | Configuration loading |

## Dependency Graph (High-Level)

```
                    ┌──────────────┐
                    │  contracts   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│     core      │  │    plugins    │  │    engine     │
│ (canonical,   │  │ (protocols,   │  │ (orchestrator,│
│  config, dag) │  │  base, impls) │  │  processor)   │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼───────┐
                    │   cli/tui    │
                    └──────────────┘
```

## Testing Strategy

| Test Type | Location | Purpose |
|-----------|----------|---------|
| Unit | `tests/core/`, `tests/plugins/` | Component isolation |
| Property | `tests/property/` | Hypothesis-based invariant testing |
| Integration | `tests/integration/` | Cross-subsystem interaction |
| System | `tests/system/` | End-to-end audit verification |
| Contract | `tests/contracts/` | Plugin contract verification |
| Example | `tests/examples/` | Example pipeline validation |

## Risk Areas for Deeper Analysis

1. **LandscapeRecorder (2,567 LOC)** - The largest file, complex state management
2. **Orchestrator (1,622 LOC)** - Complex lifecycle management
3. **Plugin Protocol interactions** - Contract enforcement
4. **Token lifecycle** - Fork/join edge cases
5. **Checkpoint/Recovery** - Resume correctness

## Existing Documentation

The codebase has excellent documentation:
- `CLAUDE.md` - Development guidelines (25K chars)
- `ARCHITECTURE.md` - C4 diagrams (420 lines)
- `PLUGIN.md` - Plugin development guide (33K chars)
- `USER_MANUAL.md` - User documentation (22K chars)
- `TEST_SYSTEM.md` - Testing philosophy (14K chars)

## Recommendations for Deep Analysis

1. **Landscape subsystem** - Critical path, needs dedicated deep-dive
2. **Engine orchestration** - Complex state machine, verify correctness
3. **Plugin schema contracts** - Ensure enforcement is complete
4. **LLM integration** - Phase 6 active, verify audit completeness
5. **Azure integration** - Phase 7 active, verify blob handling

## Confidence Assessment

| Area | Confidence | Reasoning |
|------|------------|-----------|
| Project structure | HIGH | Clear directory layout, consistent patterns |
| Subsystem boundaries | HIGH | Well-defined module exports |
| Technology stack | HIGH | pyproject.toml + code inspection |
| Dependency graph | MEDIUM | Need deeper analysis for edge cases |
| Risk areas | MEDIUM | Based on LOC and complexity, needs verification |
