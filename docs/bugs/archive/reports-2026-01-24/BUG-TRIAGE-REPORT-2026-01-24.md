# ELSPETH Bug Triage Report
**Date:** 2026-01-24
**Scope:** Complete analysis of 181 total bugs (73 open, 34 pending, 74 closed)
**Objective:** Categorize by system/subsystem, identify closure candidates, correct priorities

---

## Executive Summary

### Current State
- **73 Open Bugs**: 18 P1, 38 P2, 17 P3
- **34 Pending Bugs**: 11 P1, 16 P2, 7 P3 (18 should be promoted, 16 should be closed)
- **74 Closed Bugs**: Successfully resolved

### Critical Findings
1. **Top 3 Bug Clusters**:
   - **Audit Trail Gaps** (18 bugs): Missing payloads, outcomes, metadata
   - **Coalesce Implementation** (12 bugs): Timeouts, duplicates, late arrivals
   - **Azure LLM Integration** (14 bugs): Error handling, audit recording

2. **Immediate Action Required** (7 P0-severity bugs):
   - Data corruption in coalesce late arrivals
   - Audit integrity violation in DatabaseSink
   - Complete feature breakage (timeouts, recovery)

3. **Quality Gaps**:
   - Schema validation incomplete across pipeline
   - Type coercion violations (8 bugs) violate trust tier model
   - Error boundaries missing for external calls

---

## Proposed Categorization Taxonomy

### System → Subsystem → Component Structure

```
ELSPETH/
├── CORE/
│   ├── landscape/          # Audit trail, payload storage
│   ├── dag/                # Graph construction, validation
│   ├── config/             # Configuration loading
│   └── canonical/          # JSON canonicalization, hashing
│
├── ENGINE/
│   ├── orchestrator/       # Pipeline execution, routing
│   ├── executors/          # Transform/gate/aggregation execution
│   ├── checkpoint/         # Recovery, state persistence
│   ├── retry/              # Backoff logic, attempt tracking
│   └── schema-validation/  # Pipeline schema compatibility
│
├── PLUGINS/
│   ├── sources/            # CSV, JSON, database sources
│   ├── transforms/         # Field mapping, batch operations
│   ├── gates/              # Routing decisions
│   ├── aggregations/       # Batching, windowing, coalesce
│   └── sinks/              # CSV, database, artifact output
│
├── LLM/ (plugin pack)
│   ├── azure-batch/        # Azure batch API integration
│   ├── http/               # HTTP client, audit recording
│   └── replay/             # Record/replay/verify modes
│
├── CLI-TUI/
│   ├── cli/                # Command handling
│   └── tui/                # Terminal UI, explain queries
│
└── CROSS-CUTTING/
    ├── auditability/       # Audit completeness violations
    ├── type-coercion/      # Trust tier boundary violations
    ├── determinism/        # Reproducibility issues
    └── performance/        # N+1 queries, inefficiencies
```

---

## Subsystem Bug Breakdown

### 1. CORE System (11 bugs)

#### core/landscape (7 bugs)
- **P1 Critical** (3 bugs):
  - `P1-2026-01-22-recovery-skips-rows-multi-sink` - **FIX IMMEDIATELY**: Recovery broken for multi-sink
  - `P1-2026-01-22-run-repository-masks-invalid-export-status` - Enum validation missing
  - `P1-2026-01-22-reproducibility-grade-not-updated-after-purge` - Grade drift
- **P2** (2 bugs): Exporter missing expand group ID, rate limiter thread ID reuse
- **P3** (2 bugs): Model duplication, schema config not loaded

#### core/dag (2 bugs)
- **P1 Critical** (1 bug):
  - `P1-2026-01-22-duplicate-branch-names-break-coalesce` - **FIX IMMEDIATELY**: Silent data loss
- **P2** (1 bug): Contracts/config boundary violation

#### core/canonical (1 bug)
- **P2** (1 bug): JSONSource allows NaN/Infinity bypass

#### core/config (1 bug)
- **P2** (1 bug): Contract re-export breaks subsystem boundaries

---

### 2. ENGINE System (27 bugs) - **LARGEST CLUSTER**

#### engine/orchestrator (10 bugs)
- **P1 Critical** (2 bugs):
  - `P1-2026-01-21-orchestrator-aggregation-flush-output-mode-ignored` - **FIX IMMEDIATELY**: Token identity broken
  - `P1-2026-01-21-orchestrator-source-quarantine-outcome-missing` - Quarantine outcome not recorded
- **P2** (5 bugs): Resume edge ID mismatch, schema missing, metadata hardcoded
- **P3** (3 bugs): Checkpoint cleanup, transform close() missing

#### engine/coalesce (12 bugs) - **HIGHEST CONCENTRATION**
- **P1 Critical** (4 bugs):
  - `P1-2026-01-22-coalesce-timeouts-never-fired` - **FIX IMMEDIATELY**: Feature completely broken
  - `P1-2026-01-22-coalesce-late-arrivals-duplicate-merge` - **FIX IMMEDIATELY**: Data corruption
  - `P1-2026-01-22-coalesce-parent-outcomes-missing` - Outcomes not recorded
  - `P1-2026-01-21-token-outcome-group-ids-mismatch` - Group ID mismatch
- **P2** (8 bugs): Audit metadata, timeout failures, select fallback, duplicate overwrites

#### engine/executors (3 bugs)
- **P2** (3 bugs): Hash mismatch, missing batch_id in spans, ctx.state_id not set

#### engine/tokens (2 bugs)
- **P2** (2 bugs): Expand no deep copy, span token.id mismatch

#### engine/retry (4 bugs)
- **P2** (2 bugs): Config ignored, malformed types crash
- **P3** (2 bugs): on_retry timing, attempt index off-by-one (merge candidates)

#### engine/schema-validation (4 bugs)
- **P2** (4 bugs): Skips validation when source dynamic, ignores strict mode, type mismatches

#### engine/spans (2 bugs)
- **P2** (1 bug): Ambiguous plugin instances
- **P3** (1 bug): High cardinality span names

---

### 3. PLUGINS System (17 bugs)

#### plugins/sources (3 bugs)
- **P2** (1 bug): JSONSource missing data_key crashes
- **P3** (2 bugs): Missing plugin_version (verify OBE)

#### plugins/transforms (6 bugs)
- **P1 Critical** (2 bugs):
  - `P1-2026-01-21-batch-replicate-coerces-copies-field` - Type coercion violation
  - `P1-2026-01-21-batch-stats-skips-non-numeric-values` - Silent skip violation
- **P2** (4 bugs): Default copies validation, schema mismatches

#### plugins/sinks (8 bugs)
- **P1 Critical** (3 bugs):
  - `P1-2026-01-21-databasesink-noncanonical-hash` - **FIX IMMEDIATELY**: Audit integrity violation
  - `P1-2026-01-21-csvsink-append-schema-mismatch` - Schema ignored on append
  - `P1-2026-01-21-csvsink-mode-unvalidated-truncation` - Invalid mode truncates
- **P2** (2 bugs): Dynamic schema extra fields crash
- **P3** (3 bugs): Empty payload size cosmetic

---

### 4. LLM Integration (14 bugs) - **SECOND LARGEST CLUSTER**

#### llm/azure-batch (10 bugs)
- **P1 Critical** (3 bugs):
  - `P1-2026-01-21-azure-batch-missing-audit-payloads` - **FIX IMMEDIATELY**: Auditability violation
  - `P1-2026-01-21-azure-call-index-global` - **FIX IMMEDIATELY**: Replay broken
  - `P1-2026-01-21-azure-multi-query-batch-state-id-mismatch` - Synthetic state IDs
- **P2** (6 bugs): Error handling, JSONL parsing, schema mismatches, blob operations
- **P3** (1 bug): Blank auth fields accepted

#### llm/http (3 bugs)
- **P1** (2 bugs from pending):
  - `P1-2026-01-21-http-auth-headers-dropped-request-hash` - Audit integrity
  - `P1-2026-01-21-http-response-truncation-audit-loss` - Response truncation
- **P3** (1 bug): Missing model metadata

#### llm/replay (1 bug)
- **P1** (1 bug from pending):
  - `P1-2026-01-21-replay-request-hash-collisions` - Duplicate requests collapse

---

### 5. CLI/TUI System (3 bugs)

#### cli (2 bugs)
- **P1** (1 bug from pending):
  - `P1-2026-01-20-cli-explain-is-placeholder` - Core feature missing
- **P3** (1 bug): Enum status printing

#### tui (1 bug)
- **P3** (1 bug): Purge silently creates DB

---

### 6. CROSS-CUTTING (6 bugs)

#### auditability (covered in subsystems above)

#### type-coercion (8 bugs distributed across plugins)

#### performance (2 bugs)
- **P2** (1 bug from pending): Exporter N+1 queries

#### config/metadata (3 bugs)
- **P2** (1 bug): Hardcoded metadata in config gates
- **P3** (2 bugs): Schema hash missing, legacy shim

---

## Bug Pattern Analysis

### Top 5 Patterns (by frequency)

1. **Missing Audit Recording** (18 bugs - 24%)
   - External call payloads not stored
   - Outcomes/metadata not persisted
   - Violates "full request/response recorded" standard

2. **Schema Validation Incomplete** (11 bugs - 15%)
   - Validator skips checks
   - Type mismatches ignored
   - Extra fields crash instead of validate

3. **Type Coercion Violations** (8 bugs - 11%)
   - Transforms coercing pipeline data
   - Violates tier 2 trust model
   - BatchReplicate, BatchStats, Azure transforms

4. **Coalesce Semantic Bugs** (12 bugs - 16%)
   - Timeout never fires
   - Late arrivals duplicate
   - Parent outcomes missing

5. **Error Boundary Gaps** (6 bugs - 8%)
   - External calls not wrapped
   - Crashes instead of errors
   - Azure API, JSON parsing

---

## Priority-Based Action Plan

### **IMMEDIATE (This Week) - 7 P0-Severity Bugs**

These violate audit integrity, cause data corruption, or break core features completely:

1. `P1-2026-01-21-databasesink-noncanonical-hash`
   **Subsystem:** plugins/sinks
   **Impact:** Audit integrity violation - hashes non-canonical JSON
   **Fix:** Use canonical_json() before hashing

2. `P1-2026-01-21-azure-batch-missing-audit-payloads`
   **Subsystem:** llm/azure-batch
   **Impact:** Auditability violation - batch JSONL not recorded
   **Fix:** Store JSONL in payload store with references

3. `P1-2026-01-22-coalesce-timeouts-never-fired`
   **Subsystem:** engine/coalesce
   **Impact:** Feature completely broken - check_timeouts() never called
   **Fix:** Add timeout checking to orchestrator main loop

4. `P1-2026-01-22-coalesce-late-arrivals-duplicate-merge`
   **Subsystem:** engine/coalesce
   **Impact:** Data corruption - late arrivals create duplicate outputs
   **Fix:** Track merged state, reject late arrivals after merge

5. `P1-2026-01-21-orchestrator-aggregation-flush-output-mode-ignored`
   **Subsystem:** engine/orchestrator
   **Impact:** Token identity broken - passthrough mode doesn't preserve tokens
   **Fix:** Honor output_mode in flush logic

6. `P1-2026-01-22-duplicate-branch-names-break-coalesce`
   **Subsystem:** core/dag
   **Impact:** Silent data loss - duplicate names overwrite branches
   **Fix:** Validate unique branch names in config

7. `P1-2026-01-22-recovery-skips-rows-multi-sink`
   **Subsystem:** core/landscape
   **Impact:** Recovery broken - skips rows in multi-sink pipelines
   **Fix:** Compute recovery boundary per sink, use max sequence

---

### **HIGH PRIORITY (This Sprint) - 22 Bugs**

#### P1 Audit Trail Gaps (11 bugs)
- Azure call index global (replay broken)
- Azure multi-query state ID mismatch
- HTTP auth headers dropped
- HTTP response truncation
- LLM response partial recording
- Replay request hash collisions
- Orchestrator source quarantine outcome
- Token outcome group ID mismatch
- BatchReplicate type coercion
- BatchStats silent skip
- CSVSink append/mode bugs

#### P2 Critical Functionality (11 bugs)
- Coalesce audit metadata missing (4 bugs)
- Schema validation suite (4 bugs)
- Aggregation trigger bugs (3 bugs)

---

### **MEDIUM PRIORITY (Next Sprint) - 24 P2 Bugs**

Focus areas:
- Azure error handling suite (6 bugs)
- Sink schema handling (3 bugs)
- Transform type issues (3 bugs)
- Engine executor bugs (3 bugs)
- Exporter issues (2 bugs)
- Others (7 bugs)

---

### **LOW PRIORITY (Backlog) - 17 P3 Bugs**

Most are cosmetic or low-impact:
- **CLOSE IMMEDIATELY**: `P3-2026-01-15-defensive-whitelist-review` (stale)
- **VERIFY OBE**: Several may be fixed by recent schema validation work
- **KEEP AS TECH DEBT**: Legacy shims, cleanup tasks

---

## Pending Bugs Triage (34 bugs)

### **PROMOTE TO OPEN** (18 bugs)

#### P1 Bugs (11 bugs)
1. `cli-explain-is-placeholder` - Core feature missing
2. `aggregation-passthrough-failure-buffered` - Non-terminal tokens
3. `aggregation-single-skips-downstream` - Pipeline ordering
4. `call-index-collisions-across-clients` - Audit integrity
5. `http-auth-headers-dropped-request-hash` - Hash collisions
6. `http-response-truncation-audit-loss` - Audit loss
7. `llm-response-partial-recording` - Incomplete recording
8. `replay-request-hash-collisions` - Replay broken
9. `token-outcome-group-id-mismatch` - Correlation broken
10. `aggregation-condition-trigger-missing-row-context` - Feature broken
11. `aggregation-timeout-idle-never-fires` - Timeout broken

#### P2 Bugs (7 bugs)
- Exporter config/performance (2 bugs)
- Plugin gate graph mismatch (1 bug)
- Aggregation coalesce/config (2 bugs)
- Expression parser (2 bugs)

---

### **CLOSE AS OBE** (8 bugs)

Superseded by recent work (checkpoint fixes, schema validation architecture):
- `llm-usage-missing-crash`
- `retryable-transform-result-ignored`
- `aggregation-timeout-checkpoint-age-reset`
- `is-operator-not-restricted-to-none`
- `row-get-attribute-allowed`
- `subscript-not-restricted-to-row`
- (2 more)

**Reasoning:** Recent commits (36e17f2 checkpoint format, schema validation refactor) addressed these issues or made them obsolete.

---

### **CLOSE AS LOST** (8 bugs)

Insufficient evidence, experimental features, or design questions:
- `pooling-ordering-metadata-dropped` - Pooling is experimental
- `pooling-throttle-dispatch-burst` - Design question
- `pooling-concurrent-execute-batch-mixes-results` - Experimental
- `pooling-delay-invariant-not-validated` - Experimental
- `pooling-missing-pool-stats` - Experimental
- `verifier-missing-payload-hidden` - Edge case in optional feature
- `trigger-type-priority-misreports-first-fire` - Low impact
- `verifier-ignore-order-hides-drift` - Design choice

**Reasoning:** Pooling is experimental, verifier is optional, unclear production impact.

---

## Recommended Organizational Structure

### Proposed Directory Layout

```
docs/bugs/
├── README.md                           # Bug tracking process
├── BUG-TEMPLATE.md                     # Standard template
├── TRIAGE-GUIDELINES.md                # Triage process
│
├── open/                               # Active bugs
│   ├── by-priority/                    # Symlinks by priority
│   │   ├── P0-critical/                # Immediate action
│   │   ├── P1-high/                    # This sprint
│   │   ├── P2-medium/                  # Next sprint
│   │   └── P3-low/                     # Backlog
│   │
│   └── by-subsystem/                   # Symlinks by subsystem
│       ├── core-landscape/
│       ├── core-dag/
│       ├── engine-orchestrator/
│       ├── engine-coalesce/
│       ├── plugins-sources/
│       ├── plugins-transforms/
│       ├── plugins-sinks/
│       ├── llm-azure-batch/
│       ├── llm-http/
│       └── cross-cutting/
│
├── closed/                             # Resolved bugs
│   └── by-release/                     # Organized by release
│       ├── RC-1/
│       ├── RC-2/
│       └── archive/
│
├── pending/                            # Under investigation
│
└── metrics/                            # Bug metrics tracking
    ├── weekly-status.md
    └── bug-velocity.md
```

### File Naming Convention

**Format:** `{priority}-{date}-{subsystem}-{short-description}.md`

**Examples:**
- `P0-2026-01-24-engine-coalesce-timeouts-never-fired.md`
- `P1-2026-01-24-llm-azure-batch-missing-audit-payloads.md`
- `P2-2026-01-24-plugins-sinks-databasesink-noncanonical-hash.md`

**Benefits:**
- Sort alphabetically by priority
- Chronological within priority
- Subsystem visible at a glance
- Descriptive for quick scanning

---

## Execution Plan

### Phase 1: Immediate Triage (Today)

1. **Move pending → open** (18 bugs)
   ```bash
   mv docs/bugs/pending/P1-2026-01-20-cli-explain-is-placeholder.md \
      docs/bugs/open/P1-2026-01-20-cli-explain-is-placeholder.md
   # ... repeat for all 18 promoted bugs
   ```

2. **Close OBE bugs** (8 bugs)
   ```bash
   for bug in llm-usage-missing-crash retryable-transform-result-ignored ...; do
     echo "## Resolution: Overtaken by Events (OBE)" >> docs/bugs/pending/$bug.md
     mv docs/bugs/pending/$bug.md docs/bugs/closed/
   done
   ```

3. **Close lost bugs** (8 bugs)
   ```bash
   # Similar process with "Resolution: Closed as Lost" marker
   ```

---

### Phase 2: Renaming & Reorganization (This Week)

1. **Rename all bugs** to follow new convention
2. **Create subsystem directories** under `open/by-subsystem/`
3. **Create symlinks** for dual organization (priority + subsystem)
4. **Update BUGS.md** with categorization index

---

### Phase 3: Fix P0 Bugs (This Week)

Target the 7 P0-severity bugs in order:
1. DatabaseSink canonical hash (30 min fix)
2. Coalesce timeouts (2-3 hours)
3. Azure batch payloads (4-6 hours)
4. Coalesce late arrivals (3-4 hours)
5. Orchestrator flush mode (2-3 hours)
6. Duplicate branch names (1-2 hours)
7. Recovery multi-sink (4-6 hours)

**Estimated effort:** 2-3 days of focused work

---

### Phase 4: Systematic Bug Burndown (Sprint Planning)

**Week 1:** P0 bugs + top 5 P1 audit gaps
**Week 2:** Remaining P1 audit gaps + coalesce suite
**Week 3:** P2 Azure error handling + schema validation
**Week 4:** P2 remaining + selective P3 cleanup

---

## Metrics & Tracking

### Proposed Metrics Dashboard

Track weekly:
- **Bug Velocity**: Bugs opened vs closed
- **Subsystem Health**: Open bugs per subsystem
- **Pattern Trends**: Audit gaps, type coercion, error boundaries
- **Age Distribution**: How long bugs remain open

### Success Criteria (RC-2 Release)

- ✅ **Zero P0 bugs** (data corruption, audit integrity)
- ✅ **< 5 P1 bugs** (critical functionality)
- ✅ **< 15 P2 bugs** (major issues)
- ✅ **Audit completeness**: All external calls recorded
- ✅ **Type safety**: No coercion in transforms/sinks

---

## Appendix: Full Bug Inventory

### Open Bugs by Subsystem (73 total)

*[See subsystem breakdown sections above for detailed lists]*

### Pending Bugs Actions (34 total)

**Promote:** 18 bugs
**Close OBE:** 8 bugs
**Close Lost:** 8 bugs

### Closed Bugs (74 total)

Successfully resolved across RC-1 and ongoing RC-2 work.

---

## Recommendations Summary

### Immediate Actions (Today)
1. ✅ Triage pending bugs (18 promote, 16 close)
2. ✅ Identify P0 bugs for immediate fix (7 bugs)
3. ✅ Create subsystem categorization structure

### Short-term Actions (This Week)
1. Fix 7 P0-severity bugs
2. Rename bugs to new convention
3. Create subsystem directories
4. Update tracking documentation

### Medium-term Actions (This Sprint)
1. Systematic P1 bug burndown (22 bugs)
2. Address audit completeness gaps
3. Fix coalesce implementation suite
4. Azure integration hardening

### Long-term Actions (Next Sprint)
1. Schema validation completeness
2. Type coercion elimination
3. Error boundary hardening
4. Performance optimization (N+1 queries)

---

**Report Prepared By:** Claude Code (Exploratory Analysis)
**Review Status:** Ready for user review and action
**Next Steps:** User approval → Execute Phase 1 triage
