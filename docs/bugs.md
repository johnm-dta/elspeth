# Open Bugs - Organized by Subsystem

This directory contains all open bugs organized by the subsystem they affect. This structure enables:
- **Systematic fixing**: Tackle related bugs together
- **Developer assignment**: Assign subsystem experts to relevant bugs
- **Quality insights**: See where bugs cluster

## Current Snapshot (2026-02-13)

- Active open bug files in this directory: **1**
- Remaining open ticket:
  - `core-landscape/P2-2026-02-05-operations-i-o-payloads-lack-hashes-breaking.md`
- Notes:
  - Tickets re-verified as already fixed or overtaken were archived to `docs/bugs/closed/`.
  - Historical counts below are retained for audit history and may not reflect this current snapshot.

## Directory Structure

```
open/
├── cli/                      # Command-line interface (0 P1, 2 P2, 1 P3)
├── contracts/                # Contract validation (5 P1, 11 P2, 0 P3)
├── core-canonical/           # Canonicalization (0 P1, 1 P2, 0 P3)
├── core-checkpoint/          # Checkpointing/recovery (2 P1, 0 P2, 0 P3)
├── core-config/              # Configuration system (1 P1, 2 P2, 0 P3)
├── core-dag/                 # DAG validation, graph construction (1 P1, 0 P2, 0 P3)
├── core-landscape/           # Audit trail, recovery, verifier (10 P1, 6 P2, 0 P3)
├── core-logging/             # Logging/telemetry output (0 P1, 0 P2, 1 P3)
├── core-rate-limit/          # Rate limiters (0 P1, 1 P2, 0 P3)
├── core-retention/           # Retention/purge (1 P1, 0 P2, 0 P3)
├── core-security/            # Secret handling (1 P1, 0 P2, 0 P3)
├── cross-cutting/            # Multi-subsystem issues (EMPTY after cleanup)
├── engine-coalesce/          # Fork/join/merge logic (1 P1, 0 P2, 0 P3)
├── engine-executors/         # Executor flow (2 P1, 1 P2, 0 P3)
├── engine-expression-parser/ # Expression parsing (0 P1, 1 P2, 0 P3)
├── engine-orchestrator/      # Pipeline execution, routing (4 P1, 2 P2, 0 P3)
├── engine-pooling/           # Pooling infrastructure (1 P1, 4 P2, 2 P3)
├── engine-processor/         # Token management, outcomes (1 P1, 0 P2, 0 P3)
├── engine-retry/             # Retry logic (EMPTY after cleanup)
├── engine-spans/             # Observability, tracing (0 P1, 7 P2, 3 P3)
├── engine-tokens/            # Token lineage (0 P1, 1 P2, 0 P3)
├── engine-triggers/          # Trigger evaluation (1 P1, 0 P2, 0 P3)
├── mcp/                      # MCP tooling (1 P1, 1 P2, 0 P3)
├── plugins-azure/            # Azure plugin pack (3 P1, 3 P2, 0 P3)
├── plugins-llm/              # Base LLM transforms (6 P1, 11 P2, 2 P3)
├── plugins-sinks/            # Sink implementations (3 P1, 0 P2, 1 P3)
├── plugins-sources/          # Source implementations (1 P1, 6 P2, 1 P3)
├── plugins-transforms/       # Transform implementations (6 P1, 16 P2, 2 P3)
└── testing/                  # Testing infrastructure (EMPTY after cleanup)
```

## Bug Counts by Subsystem

| Subsystem | P1 Bugs | P2 Bugs | P3 Bugs | Total | Notes |
|-----------|---------|---------|---------|-------|-------|
| **plugins-transforms** | 6 | 16 | 2 | 24 | Transform config, field mapping, batch stats |
| **plugins-llm** | 6 | 11 | 2 | 19 | Azure/OpenRouter transforms, HTTP clients |
| **contracts** | 5 | 11 | 0 | 16 | Schema contracts, type normalization |
| **core-landscape** | 10 | 6 | 0 | 16 | Audit trail, repositories, payload store |
| **engine-spans** | 0 | 7 | 3 | 10 | Telemetry, exporters |
| **plugins-sources** | 1 | 6 | 1 | 8 | CSV/JSON sources, field normalization |
| **engine-pooling** | 1 | 4 | 2 | 7 | Batching, pooling infrastructure |
| **plugins-azure** | 3 | 3 | 0 | 6 | Azure blob source/sink |
| **engine-orchestrator** | 4 | 2 | 0 | 6 | Resume, aggregation, export |
| **plugins-sinks** | 3 | 0 | 1 | 4 | CSV/JSON/Database sinks |
| **cli** | 0 | 2 | 1 | 3 | CLI commands |
| **core-config** | 1 | 2 | 0 | 3 | Configuration, gate normalization |
| **engine-executors** | 2 | 1 | 0 | 3 | Transform/gate/sink execution |
| **core-checkpoint** | 2 | 0 | 0 | 2 | Checkpoint serialization, recovery |
| **mcp** | 1 | 1 | 0 | 2 | MCP server |
| **core-canonical** | 0 | 1 | 0 | 1 | Canonical JSON |
| **core-dag** | 1 | 0 | 0 | 1 | DAG validation |
| **core-logging** | 0 | 0 | 1 | 1 | Logging |
| **core-rate-limit** | 0 | 1 | 0 | 1 | Rate limiting |
| **core-retention** | 1 | 0 | 0 | 1 | Payload purge |
| **core-security** | 1 | 0 | 0 | 1 | Secret loading |
| **engine-coalesce** | 1 | 0 | 0 | 1 | Fork/join execution |
| **engine-expression-parser** | 0 | 1 | 0 | 1 | Expression parsing |
| **engine-processor** | 1 | 0 | 0 | 1 | Token processing |
| **engine-tokens** | 0 | 1 | 0 | 1 | Token management |
| **engine-triggers** | 1 | 0 | 0 | 1 | Trigger evaluation |
| **TOTAL** | **51** | **76** | **13** | **140** | After 2026-02-05 validation |

## Priority Distribution

| Priority | Count | Percentage | Description |
|----------|-------|------------|-------------|
| **P1** | 51 | 36% | Major bugs - audit/data integrity issues |
| **P2** | 76 | 54% | Moderate bugs - functionality gaps |
| **P3** | 13 | 9% | Minor bugs - quality/UX issues |

## 2026-02-05 Validation Summary

**Changes made during validation:**
- **Removed 64 false positives**: Bug tickets with "no bug found" or "no concrete bug found" in analysis
- **Downgraded 3 P1 bugs to P2**: Severity was inflated (functionality gaps, not audit integrity)
  - `pluginconfigvalidator-rejects-openrouter-batc` - Validator missing plugins (P1 -> P2)
  - `openrouter-batch-drops-api-v1-from-base-ur` - Wrong endpoint (P1 -> P2)
  - `nullsourceschema-treated-as-explicit-schema` - Resume validation (P1 -> P2)
- **No duplicates found**: Verified against docs/bugs/closed/

**Bug count changes:**
- Original from triage: 203 bugs
- After removing false positives: 139 bugs
- After severity corrections: 140 bugs (51 P1, 76 P2, 13 P3)

## Recommended Fix Order

### Phase 1: Critical Audit/Data Integrity (P1 - Fix This Sprint)

**Hotspots requiring immediate attention:**
- **core-landscape/** (10 P1): Repository validation, payload persistence, reproducibility grading
- **plugins-transforms/** (6 P1): Field mapping contracts, capacity errors, config validation
- **plugins-llm/** (6 P1): HTTP/LLM call recording, NaN validation, output key collisions
- **contracts/** (5 P1): Schema contract locking, type normalization, routing validation
- **engine-orchestrator/** (4 P1): Resume contract inference, quarantine hashing, aggregation triggers

### Phase 2: Functional Gaps (P2 - Fix Next Sprint)

**Focus areas:**
- **plugins-transforms/** (16 P2): Config validation, contract propagation
- **contracts/** (11 P2): Schema handling, header modes, error contracts
- **plugins-llm/** (11 P2): Batch transforms, tracing, response validation
- **engine-spans/** (7 P2): Telemetry filtering, exporter configuration

### Phase 3: Quality & Polish (P3 - Backlog)

- Observability improvements (engine-spans)
- CLI error messages
- Minor validation gaps
- UX improvements

## Subsystem Ownership

### CORE Subsystem
**Owner:** Systems architect
**Files:** `src/elspeth/core/`
- `landscape/` - Repositories, recorder, exporter, verifier
- `checkpoint/` - Resume, checkpoint serialization
- `dag.py` - Schema contract propagation
- `payload_store.py` - Integrity checking
- `rate_limit/` - Thread safety
- `security/` - Key vault integration

### ENGINE Subsystem
**Owner:** Pipeline execution expert
**Files:** `src/elspeth/engine/`
- `orchestrator/` - Resume contract handling, aggregation
- `executors.py` - Contract propagation, hash error handling
- `coalesce_executor.py` - Late arrival handling
- `processor.py` - Deaggregation with PipelineRow
- `triggers.py` - Condition latching

### PLUGINS Subsystem
**Owner:** Plugin developer
**Files:** `src/elspeth/plugins/`
- `llm/` - Azure/OpenRouter transforms, multi-query
- `transforms/` - FieldMapper contracts, batch stats
- `sinks/` - CSV/JSON/Database output handling
- `sources/` - CSV/JSON parsing, field normalization
- `azure/` - Blob source/sink handling

### CONTRACTS Subsystem
**Owner:** Data architect
**Files:** `src/elspeth/contracts/`
- Schema contracts, type normalization
- Routing contracts, audit contracts
- Transform output contracts

## Reports

- **Validation Report:** `docs/bugs/BUG-VALIDATION-REPORT-2026-02-05.md`
- **Original Triage Report:** `docs/bugs/BUG-TRIAGE-REPORT-2026-02-05.md`
- **Historical verification:** `docs/bugs/VERIFICATION-REPORT-2026-01-25.md`
