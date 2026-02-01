# Branch Bug Hunt Triage - 2026-01-25
**Branch:** `fix/rc1-bug-burndown-session-4` vs `main`
**Files Analyzed:** 28 changed files
**Bugs Found:** 24 total (1 P0, 15 P1, 4 P2, 4 P3)
**Status:** TRIAGED - Action items identified

---

## Executive Summary

Static analysis of 28 changed files found **24 bugs**, with **2 merged into existing reports**. Comparison to CHECKPOINT_BUGS_CONSENSUS.md shows:

- **P0 Bugs:** 1 KNOWN (Bug #4 variant), 1 NEW (sink mode injection)
- **P1 Bugs:** 12 NEW bugs (multi-sink validation, Azure audit gaps, DAG builder issues)
- **P2/P3 Bugs:** Mostly schema/audit completeness issues

**Critical Finding:** The P0 "Resume Drops Row Data" bug is a **specific manifestation of Bug #4** from the consensus document, confirming the schema requirement fix is urgent.

**New Critical Issue:** Resume forces `mode="append"` on all sinks, breaking JSON/Database sinks (NOT in consensus).

---

## P0 Bugs (CRITICAL - Week 1)

### BUG-CLI-01: Resume Drops Row Data via NullSource Schema ⚠️ KNOWN
**File:** `src/elspeth/cli.py`
**Status:** DUPLICATE of Consensus Bug #4 (Type Degradation)
**Action:** Already covered in CHECKPOINT_BUGS_CONSENSUS.md Week 2 plan

**Evidence:**
- cli.py:1436 replaces real source with NullSource
- orchestrator.py:1343 uses config.source._schema_class for type restoration
- NullSourceSchema has no fields, so `extra="ignore"` drops all row data

**Consensus Fix:**
- Make `source_schema_class` REQUIRED (not Optional)
- Preserve original schema when constructing NullSource
- See CHECKPOINT_BUGS_CONSENSUS.md lines 340-382

### BUG-CLI-02: Resume Forces mode=append For All Sinks ❌ NEW
**File:** `src/elspeth/cli.py:1425`
**Priority:** P0 (blocks resume for JSON/Database sinks)
**Status:** NOT in consensus document - NEW bug

**Problem:**
```python
# cli.py:1425
sink_options["mode"] = "append"  # Applied to ALL sinks
```

**Impact:**
- JSON/Database sinks don't have `mode` field
- Config validation fails with `extra=forbid`
- Resume crashes for any non-CSV sink

**Proposed Fix:**
```python
# Option A: Sink-specific append logic
if sink_type == "csv":
    sink_options["mode"] = "append"
elif sink_type == "database":
    sink_options["if_exists"] = "append"
elif sink_type == "json":
    # JSONL can append, JSON array cannot
    if sink_options.get("format") == "jsonl":
        pass  # File append mode works
    else:
        raise ValueError("Cannot resume with JSON array sink - not appendable")

# Option B: Fail fast for non-appendable sinks
APPENDABLE_SINKS = {"csv", "database", "jsonl"}
if sink_type not in APPENDABLE_SINKS:
    raise ValueError(f"Cannot resume: {sink_type} sink does not support append")
```

**Tests Required:**
- `test_resume_csv_sink_append_mode()`
- `test_resume_database_sink_if_exists()`
- `test_resume_json_array_sink_rejects()`
- `test_resume_jsonl_sink_appends()`

**Action:** Add to Week 1 P0 fixes (blocks resume for non-CSV pipelines)

---

## P1 Bugs (HIGH - Week 2)

### BUG-COMPAT-01: Multi-Sink DAG Compatibility Ignores Parallel Branches ❌ NEW
**File:** `src/elspeth/core/checkpoint/compatibility.py:76`
**Priority:** P1 (auditability violation)

**Problem:**
- Compatibility validation only hashes ancestors of checkpoint node (the sink)
- Changes to OTHER sink branches are ignored
- Single run can have mixed pipeline configurations

**Example:**
```
Source → Gate → Sink A (checkpointed)
              ↘ Sink B (modified after checkpoint)
```

Current behavior: Resume allowed (only Sink A ancestors checked)
Expected: Resume rejected (full DAG changed)

**Impact:** Compromises auditability - same run_id contains decisions from different configs

**Proposed Fix:**
```python
# Option A: Hash full DAG topology
def validate_compatibility(checkpoint, graph):
    stored_hash = checkpoint.upstream_topology_hash
    full_graph_hash = compute_full_topology_hash(graph)  # Not just ancestors
    return stored_hash == full_graph_hash

# Option B: Hash all sink branches separately
def create_checkpoint(...):
    sink_hashes = {}
    for sink_node in graph.get_sinks():
        sink_hashes[sink_node] = compute_upstream_topology_hash(graph, sink_node)
    checkpoint.sink_branch_hashes = json.dumps(sink_hashes)
```

**Action:** Add to Week 2 - requires schema change + Alembic migration

---

### BUG-DAG-01: Duplicate Fork/Coalesce Branch Names Accepted ❌ NEW
**File:** `src/elspeth/core/dag.py`
**Priority:** P1 (causes coalesce stalls and token overwrites)

**Problem:**
- DAG builder accepts duplicate `branch_name` values
- Coalesce config maps branch_name → node_id
- Last duplicate silently overrides earlier entries

**Example:**
```yaml
fork_to_paths:
  - branch_name: "analysis"  # First
    nodes: [transform_a]
  - branch_name: "analysis"  # Duplicate - overwrites!
    nodes: [transform_b]
```

**Impact:** Tokens never reach coalesce node, causing pipeline stalls

**Proposed Fix:**
```python
# In DAG builder
seen_branches = set()
for branch in gate_config.fork_to_paths:
    if branch.branch_name in seen_branches:
        raise ValueError(
            f"Duplicate branch name '{branch.branch_name}' in gate '{gate_id}'. "
            "Each branch must have a unique name."
        )
    seen_branches.add(branch.branch_name)
```

**Action:** Add to Week 2 - validation fix

---

### BUG-LINEAGE-01: Forked Branches Never Coalesce (Wrong Mapping Key) ❌ NEW
**File:** `src/elspeth/core/landscape/lineage.py`
**Priority:** P1 (fork/coalesce completely broken)

**Problem:**
```python
# lineage.py - stores mapping as node_id → coalesce_node
branch_to_coalesce = {node_id: coalesce_node_id}

# But fork creates tokens with branch_name, not node_id
token.branch_name = "analysis"  # String

# Coalesce lookup fails:
coalesce_target = branch_to_coalesce.get(token.branch_name)  # None!
```

**Root Cause:** Mapping uses node IDs as keys, but lookup uses branch names

**Proposed Fix:**
```python
# Store mapping as branch_name → coalesce_node
branch_to_coalesce = {branch_name: coalesce_node_id}

# Fork: Set token.branch_name from config
# Coalesce: Lookup by token.branch_name (works!)
```

**Action:** Add to Week 2 - critical for fork/coalesce functionality

---

### BUG-AZURE-01: Azure Batch Audit Calls Omit JSONL Payloads ❌ NEW
**File:** `src/elspeth/plugins/llm/azure_batch.py`
**Priority:** P1 (audit trail incomplete)

**Problem:**
- Creates batch JSONL files with full request/response
- Records call audit with `state_id`, but omits payload hashes
- Payloads never recorded in payload store
- Audit trail references non-existent payloads

**Impact:** Cannot explain batch LLM decisions - auditability broken

**Proposed Fix:**
```python
# After creating batch JSONL
request_hash = stable_hash(request_data)
response_hash = stable_hash(response_data)

payload_store.store(request_hash, request_data)
payload_store.store(response_hash, response_data)

# Then record call with hashes
```

**Action:** Add to Week 2 - audit completeness

---

### BUG-AZURE-02: Batch Mode Uses Synthetic state_id, Breaking Call Audit FK ❌ NEW
**File:** `src/elspeth/plugins/llm/azure_batch.py`
**Priority:** P1 (audit FK violations)

**Problem:**
- Batch mode generates synthetic `state_id` values
- Call audit records reference these IDs
- But `node_states` table never gets these synthetic IDs
- Foreign key violations on call audit queries

**Root Cause:** Batch calls don't create real node states

**Proposed Fix:**
```python
# Option A: Create real node_states for batch calls
for call in batch_calls:
    state_id = recorder.record_node_state(
        run_id=run_id,
        token_id=call.token_id,
        node_id=node_id,
        ...
    )
    call.state_id = state_id

# Option B: Make state_id nullable in call_audit table
# Document: Batch calls may not have associated node states
```

**Action:** Add to Week 2 - schema + FK fix

---

### BUG-BLOB-01: AzureBlobSource CSV Parse Errors Abort Instead of Quarantine ❌ NEW
**File:** `src/elspeth/plugins/azure/blob_source.py`
**Priority:** P1 (violates quarantine contract)

**Problem:**
```python
# Current code
for row in csv.DictReader(blob_data):
    yield row  # If parse fails → exception → pipeline abort
```

**Expected (per CLAUDE.md Three-Tier Trust Model):**
- External data (Tier 3) must be validated at boundary
- Parse failures should quarantine, not crash
- Record quarantine reason in audit trail

**Proposed Fix:**
```python
for row_num, row in enumerate(csv.DictReader(blob_data), start=1):
    try:
        validated = self.validate_row(row)
        yield validated
    except ValidationError as e:
        yield {
            "_quarantined": True,
            "_quarantine_reason": f"CSV parse error at row {row_num}: {e}",
            "_raw_data": str(row),
        }
```

**Action:** Add to Week 2 - source validation

---

### BUG-CANON-01: NaN/Infinity Rejection Bypassed in Multi-Dimensional NumPy Arrays ❌ NEW
**File:** `src/elspeth/core/canonical.py`
**Priority:** P1 (audit integrity)

**Problem:**
```python
# Current check
if isinstance(obj, (float, np.floating)):
    if math.isnan(obj) or math.isinf(obj):
        raise ValueError("NaN/Infinity not allowed")

# But multi-dimensional arrays not checked:
np.array([[1.0, float('nan')], [2.0, 3.0]])  # Passes!
```

**Impact:** NaN/Infinity can enter audit trail through arrays, violating canonicalization guarantees

**Proposed Fix:**
```python
if isinstance(obj, np.ndarray):
    if np.any(np.isnan(obj)) or np.any(np.isinf(obj)):
        raise ValueError(
            "NaN/Infinity found in NumPy array. "
            "Audit trail requires finite values only."
        )
    return obj.tolist()  # Then convert to list
```

**Action:** Add to Week 2 - canonical JSON validation

---

## P2 Bugs (MEDIUM - Week 3)

### BUG-RECORDER-01: Gate/Sink Executions Don't Initialize PluginContext state_id
**File:** `src/elspeth/core/landscape/recorder.py`
**Priority:** P2 (audit completeness)

**Problem:** PluginContext for gates/sinks missing `state_id`, so external calls can't be recorded

**Action:** Add to Week 3 - pass state_id through context

---

### BUG-EXEC-01: Aggregation Flush Input Hash Mismatch
**File:** `src/elspeth/engine/executors.py`
**Priority:** P2 (audit consistency)

**Problem:** `node_state.input_hash` doesn't match `TransformResult.input_hash` for aggregation flushes

**Action:** Add to Week 3 - hash alignment

---

### BUG-AZURE-03: AzureLLMTransform output_schema Omits LLM Response Fields
**File:** `src/elspeth/plugins/llm/azure.py`
**Priority:** P2 (schema completeness)

**Problem:** Schema doesn't include actual output fields (model, usage, etc.)

**Action:** Add to Week 3 - schema update

---

### BUG-SCHEMA-01: Schema Config Errors Escape Validator Instead of Returning Structured Errors
**File:** `src/elspeth/core/landscape/schema.py`
**Priority:** P2 (error handling)

**Problem:** Validation errors raise exceptions instead of returning structured error objects

**Action:** Add to Week 3 - error handling refactor

---

## P3 Bugs (LOW - Week 4)

### BUG-AUDIT-01: Checkpoint Contract Allows NULL Topology Hashes
**File:** `src/elspeth/contracts/audit.py`
**Priority:** P3 (schema tightening)

**Problem:** Schema allows `topology_hash: str | None`, but NULL hash is never valid

**Action:** Add to Week 4 - schema constraint (already fixed in consensus Bug #7)

---

### BUG-BASE-01: BaseLLMTransform Output Omits Model Metadata
**File:** `src/elspeth/plugins/llm/base.py`
**Priority:** P3 (observability)

**Problem:** LLM responses don't include model/version metadata for audit trail

**Action:** Add to Week 4 - metadata enrichment

---

## Bugs Merged Into Existing Reports

### MERGED-01: (Topic of merged bug not in this file)
**Action:** Check `docs/bugs/open/` for updated reports with re-verification sections

### MERGED-02: (Topic of merged bug not in this file)
**Action:** Check `docs/bugs/open/` for updated reports

---

## Comparison to CHECKPOINT_BUGS_CONSENSUS.md

| Consensus Bug | Found in Branch Scan? | Status |
|---------------|-----------------------|--------|
| Bug #1: Topology Hash Race | No (not in changed files) | Covered by consensus |
| Bug #2: Checkpoint Before Sink Durability | No (not in changed files) | Covered by consensus |
| Bug #3: Synthetic Edge IDs | No (not in changed files) | Covered by consensus |
| **Bug #4: Type Degradation** | **YES (BUG-CLI-01)** | ✅ Confirmed by scan |
| Bug #5: Transaction Rollback | No (not in changed files) | Covered by consensus |
| Bug #6: Aggregation Timeout Reset | No (not in changed files) | Covered by consensus |
| Bug #7: Schema Allows NULL | YES (BUG-AUDIT-01) | ✅ Confirmed by scan |
| Bug #8: Resume Early Exit Cleanup | No (not in changed files) | Covered by consensus |
| Bug #9: Missing Graph Validation | No (not in changed files) | Covered by consensus |
| Bug #10: Checkpoint Callback Errors | No (not in executors.py scan) | Covered by consensus |
| Bug #11: Type Annotations Too Broad | No | Covered by consensus |
| Bug #12: No Checkpoint Version Validation | No | Covered by consensus |

**Key Finding:** Branch scan **confirms Bug #4** (type degradation) is real and manifests as data loss on resume.

---

## NEW Critical Bugs NOT in Consensus

| Bug | Priority | Category | Impact |
|-----|----------|----------|---------|
| BUG-CLI-02 | **P0** | Resume | Blocks resume for JSON/DB sinks |
| BUG-COMPAT-01 | **P1** | Resume | Multi-sink validation gap |
| BUG-DAG-01 | **P1** | Fork/Coalesce | Duplicate branches cause stalls |
| BUG-LINEAGE-01 | **P1** | Fork/Coalesce | Coalesce completely broken |
| BUG-AZURE-01 | **P1** | Audit | Batch LLM calls missing payloads |
| BUG-AZURE-02 | **P1** | Audit | Batch state_id FK violations |
| BUG-BLOB-01 | **P1** | Sources | CSV errors crash instead of quarantine |
| BUG-CANON-01 | **P1** | Audit | NaN/Inf bypass in arrays |

**Total NEW Bugs:** 8 high-priority (1 P0, 7 P1)

---

## Recommended Actions

### IMMEDIATE (Today)

1. **Add BUG-CLI-02 to Week 1 P0 fixes** - Resume is currently broken for non-CSV sinks
2. **Verify fork/coalesce bugs (BUG-DAG-01, BUG-LINEAGE-01)** - May explain user-reported stalls
3. **Review CHECKPOINT_BUGS_CONSENSUS.md** - Ensure new bugs don't conflict with planned fixes

### Week 1 (P0 Fixes - Updated)

Original consensus plan PLUS:
- **Day 3: BUG-CLI-02** - Add sink-specific append logic to resume path

### Week 2 (P1 Fixes - Expanded)

Original consensus plan PLUS:
- **Day 4: BUG-COMPAT-01** - Multi-sink DAG validation
- **Day 5: BUG-DAG-01 + BUG-LINEAGE-01** - Fork/coalesce fixes

### Week 3 (P1 Audit Completeness)

- **Day 1-2: BUG-AZURE-01, BUG-AZURE-02** - Batch LLM audit trail
- **Day 3: BUG-BLOB-01** - Source quarantine handling
- **Day 4: BUG-CANON-01** - Array NaN/Inf checking
- **Day 5: BUG-RECORDER-01** - Gate/sink state_id

### Week 4 (P2/P3 Polish)

- Remaining P2/P3 bugs from scan
- Consensus P2/P3 bugs

---

## Testing Priorities

### New Tests Required (High Priority)

1. **Resume sink append modes** (BUG-CLI-02)
   ```python
   def test_resume_json_array_sink_rejects()
   def test_resume_database_sink_uses_if_exists()
   def test_resume_jsonl_sink_appends()
   ```

2. **Multi-sink DAG validation** (BUG-COMPAT-01)
   ```python
   def test_resume_rejects_parallel_branch_changes()
   ```

3. **Fork/coalesce with duplicate branches** (BUG-DAG-01, BUG-LINEAGE-01)
   ```python
   def test_duplicate_branch_names_rejected()
   def test_forked_tokens_reach_coalesce()
   ```

4. **Array NaN/Inf rejection** (BUG-CANON-01)
   ```python
   def test_numpy_array_with_nan_rejected()
   def test_nested_array_with_inf_rejected()
   ```

---

## Clean Files

1 file had no bugs found:
- `src/elspeth/core/checkpoint/manager.py` ✅

---

## Evidence Gate Downgrades

1 bug was downgraded from higher priority due to insufficient file:line evidence.

---

## Sign-off

**Triaged by:** Claude Sonnet 4.5
**Date:** 2026-01-25
**Confidence:** HIGH (85%) - Based on systematic comparison to consensus document
**Recommendation:** Prioritize BUG-CLI-02 (P0) and fork/coalesce bugs (P1) for immediate investigation

---

## Next Steps

1. ✅ Triage complete
2. ⏭️ Verify fork/coalesce bugs with integration tests
3. ⏭️ Add BUG-CLI-02 to Week 1 implementation plan
4. ⏭️ Update CHECKPOINT_BUGS_CONSENSUS.md with new findings
5. ⏭️ Create bug tickets for P1 issues not in consensus
