# Open Bugs - Organized by Subsystem

This directory contains all open bugs organized by the subsystem they affect. This structure enables:
- **Systematic fixing**: Tackle related bugs together
- **Developer assignment**: Assign subsystem experts to relevant bugs
- **Quality insights**: See where bugs cluster

## Directory Structure

```
open/
├── cli/                      # Command-line interface (0 P1, 0 P2, 0 P3)
├── contracts/                # Contract validation (0 P1, 0 P2, 0 P3)
├── core-canonical/           # Canonicalization (0 P1, 0 P2, 0 P3)
├── core-checkpoint/          # Checkpointing/recovery (0 P1, 0 P2, 0 P3)
├── core-config/              # Configuration system (0 P1, 0 P2, 0 P3)
├── core-dag/                 # DAG validation, graph construction (0 P1, 0 P2, 0 P3)
├── core-landscape/           # Audit trail, recovery, verifier (0 P1, 0 P2, 0 P3)
├── core-logging/             # Logging/telemetry output (0 P1, 0 P2, 0 P3)
├── core-rate-limit/          # Rate limiters (0 P1, 0 P2, 0 P3)
├── core-retention/           # Retention/purge (0 P1, 0 P2, 0 P3)
├── core-security/            # Secret handling (0 P1, 0 P2, 0 P3)
├── engine-coalesce/          # Fork/join/merge logic (0 P1, 0 P2, 0 P3) - EMPTY
├── engine-executors/         # Executor flow (0 P1, 0 P2, 0 P3)
├── engine-expression-parser/ # Expression parsing (0 P1, 0 P2, 0 P3)
├── engine-orchestrator/      # Pipeline execution, routing (0 P1, 0 P2, 0 P3)
├── engine-pooling/           # Pooling infrastructure (0 P1, 1 P2, 1 P3)
├── engine-processor/         # Token management, outcomes (0 P1, 0 P2, 0 P3)
├── engine-retry/             # Retry logic (0 P1, 0 P2, 0 P3)
├── engine-spans/             # Observability, tracing (0 P1, 0 P2, 1 P3)
├── engine-tokens/            # Token lineage (0 P1, 0 P2, 0 P3)
├── engine-triggers/          # Trigger evaluation (0 P1, 0 P2, 0 P3)
├── mcp/                      # MCP tooling (0 P1, 0 P2, 0 P3)
├── plugins-azure/            # Azure plugin pack (0 P1, 0 P2, 0 P3)
├── plugins-llm/              # Base LLM transforms (0 P1, 0 P2, 0 P3)
├── plugins-sinks/            # Sink implementations (0 P1, 0 P2, 0 P3)
├── plugins-sources/          # Source implementations (0 P1, 0 P2, 0 P3)
└── plugins-transforms/       # Transform implementations (0 P1, 0 P2, 0 P3)
```

## Bug Counts by Subsystem

| Subsystem | P1 Bugs | P2 Bugs | P3 Bugs | Total | Notes |
|-----------|---------|---------|---------|-------|-------|
| **core-landscape** | 0 | 0 | 0 | 0 | — |
| **engine-orchestrator** | 0 | 0 | 0 | 0 | — |
| **contracts** | 0 | 0 | 0 | 0 | — |
| **plugins-llm** | 0 | 0 | 0 | 0 | — |
| **engine-coalesce** | 0 | 0 | 0 | 0 | Fork/join timeouts (all fixed) |
| **core-checkpoint** | 0 | 0 | 0 | 0 | — |
| **core-config** | 0 | 0 | 0 | 0 | — |
| **engine-pooling** | 0 | 1 | 1 | 2 | Pooling/batching |
| **engine-processor** | 0 | 0 | 0 | 0 | — |
| **engine-retry** | 0 | 0 | 0 | 0 | — |
| **engine-spans** | 0 | 0 | 1 | 1 | Observability spans |
| **core-rate-limit** | 0 | 0 | 0 | 0 | — |
| **plugins-sinks** | 0 | 0 | 0 | 0 | — |
| **cli** | 0 | 0 | 0 | 0 | — |
| **core-dag** | 0 | 0 | 0 | 0 | — |
| **core-payload** | 0 | 0 | 0 | 0 | Payload storage (all fixed) |
| **core-retention** | 0 | 0 | 0 | 0 | — |
| **core-security** | 0 | 0 | 0 | 0 | — |
| **engine-executors** | 0 | 0 | 0 | 0 | — |
| **mcp** | 0 | 0 | 0 | 0 | — |
| **plugins-azure** | 0 | 0 | 0 | 0 | — |
| **core-canonical** | 0 | 0 | 0 | 0 | — |
| **core-logging** | 0 | 0 | 0 | 0 | — |
| **engine-expression-parser** | 0 | 0 | 0 | 0 | — |
| **engine-tokens** | 0 | 0 | 0 | 0 | — |
| **engine-triggers** | 0 | 0 | 0 | 0 | — |
| **plugins-sources** | 0 | 0 | 0 | 0 | — |
| **plugins-transforms** | 0 | 0 | 0 | 0 | — |
| **TOTAL** | **0** | **1** | **2** | **3** | All bugs organized |

## Recommended Fix Order

### Phase 1: Critical Data Integrity (P1 - Fix This Sprint)
- None (no open P1s as of 2026-02-01).

Note: P1s closed during RC1 bug hunt:
- **engine-coalesce/P1-2026-01-30-require-all-timeout-ignored** - Fixed with require_all timeout handling
- **core-payload/P1-2026-01-31-payload-store-path-traversal** - Fixed with hash validation + containment
- **cli/P1-2026-01-31-settings-path-missing-silent-fallback** - Already fixed in commit 8ab8fb36
- **engine-orchestrator/P1-2026-01-31-quarantine-outcome-before-durability** - Fixed in commit e039498b
- **core-dag/P1-2026-01-31-gate-drops-computed-schema-guarantees** - Fixed in commit 08cec258

### Phase 2: Audit Trail Completeness (Fix This Sprint/Next)
- **core-landscape/** - Exporter completeness, verifier correctness
- **contracts/** - Schema compatibility and enum validation
- **engine-orchestrator/** - Cleanup error surfacing + output canonical validation
- **plugins-llm/** - Azure/OpenRouter call recording and retry semantics

### Phase 3: Quality & Observability (Next Sprint)
- **engine-spans/** - Span naming/metadata issues
- **engine-pooling/** - Pool stats + batching examples
- **core-logging/** - JSON log output consistency

## Subsystem Ownership

### CORE Subsystem
**Owner:** Systems architect
**Files:** `src/elspeth/core/`
- `landscape/` - export, verifier, run repository
- `checkpoint/` - resume + checkpoint format
- `dag.py` - edge compatibility + coalesce validation
- `payload_store.py` - path traversal + required payload storage
- `rate_limit/` - atomicity + stale counters

### ENGINE Subsystem
**Owner:** Pipeline execution expert
**Files:** `src/elspeth/engine/`
- `coalesce_executor.py` - timeouts + branch handling
- `orchestrator.py` - cleanup + quarantine durability ordering
- `executors.py` - config gate missing-edge handling
- `processor.py` - token span + outcome handling

### PLUGINS Subsystem
**Owner:** Plugin developer
**Files:** `src/elspeth/plugins/`
- `llm/` - multi-query output semantics, call recording
- `transforms/` - call_index allocation
- `sinks/` - schema validation, payload size issues

## Verification Status

**Open bugs (as of 2026-02-01): 0 P1, 1 P2, 2 P3 = 3 total.**

**Triage updates (2026-02-01):**
- Removed 17 open entries that already existed under `docs/bugs/closed/` (duplicates).
- Moved `P3-2026-01-21-verifier-ignore-order-hides-drift.md` to closed (Phase 1 implemented).
- Normalized four misfiled bugs into subsystem folders with standard filenames.
- Verified all remaining P1s; closed two that are now fixed (`P1-2026-01-30-payload-store-optional`, `P1-2026-01-31-sink-flush-failure-leaves-open-states`).
- Closed `P2-2026-01-22-coalesce-timeout-failures-unrecorded` (fixed coalesce failure recording).
- Closed `P3-2026-01-22-engine-artifacts-legacy-shim` (removed legacy re-export).
- Closed `P3-2026-01-31-payload-store-legacy-reexport` (removed legacy re-export).
- Closed `P3-2026-01-31-rowresult-legacy-accessors` (removed legacy accessors).

**Reports:**
- Original verification: `docs/bugs/VERIFICATION-REPORT-2026-01-25.md`
- Pending verification: `docs/bugs/VERIFICATION-REPORT-PENDING-2026-01-25.md`
