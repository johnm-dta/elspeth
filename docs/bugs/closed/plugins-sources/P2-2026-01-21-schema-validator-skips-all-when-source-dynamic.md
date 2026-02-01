# Bug Report: schema_validator skips all validation when source schema is None

## Summary

- `validate_pipeline_schemas` returns early if `source_output` is None.
- This bypasses validation for transform-to-transform and transform-to-sink compatibility even when those schemas are explicitly declared.
- As a result, common pipelines with dynamic sources get no schema validation at all.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any pipeline with dynamic source and explicit downstream schemas

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/schema_validator.py`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Set `source_output=None` (dynamic source).
2. Provide incompatible transform and sink schemas (e.g., transform outputs `result: str`, sink requires `result: str` and `extra: int`).
3. Call `validate_pipeline_schemas(...)`.

## Expected Behavior

- Validation should still check compatibility between downstream stages that have explicit schemas.

## Actual Behavior

- Validation returns no errors because it exits early.

## Evidence

- Early return on dynamic source: `src/elspeth/engine/schema_validator.py:40-44`

## Impact

- User-facing impact: pipelines with dynamic sources receive no schema validation, even when downstream schemas are strict.
- Data integrity / security impact: missing early detection of incompatible transforms/sinks.
- Performance or cost impact: errors surface at runtime instead of build time.

## Root Cause Hypothesis

- The validator treats a dynamic source as reason to skip all checks, rather than only the source-to-first-transform edge.

## Proposed Fix

- Code changes (modules/files):
  - Only skip the source->first-transform validation when source schema is None.
  - Still validate transform chain and sinks when their schemas are explicit.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test verifying downstream validation still runs with dynamic sources.
- Risks or migration steps:
  - Pipelines using dynamic sources may now fail validation if downstream schemas are incompatible.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (engine validates schema compatibility between connected nodes)
- Observed divergence: dynamic source disables all validation.
- Reason (if known): early return in validator.
- Alignment plan or decision needed: define validation scope when source is dynamic.

## Acceptance Criteria

- Downstream schema compatibility is validated even when the source schema is dynamic.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## Verification (2026-01-24)

**Status:** ✅ **OBE (Overtaken By Events) - Root Cause Fixed**

**Verifier:** Claude Sonnet 4.5 (verification task)

### Evidence of Resolution

**1. Target code deleted:**
- `src/elspeth/engine/schema_validator.py` was deleted in commit `f4dd59d` (2026-01-24)
- The `validate_pipeline_schemas()` function referenced in this bug report no longer exists
- Test files `tests/engine/test_schema_validator.py` and `tests/engine/test_orchestrator_schema_validation.py` were also deleted

**2. Architecture completely refactored (2x):**

**First refactor (commit f4dd59d, 2026-01-24):**
- Moved schema validation from standalone module into `ExecutionGraph` class
- Validation became edge-based (DAG-aware) instead of linear list-based
- Fixed P1-2026-01-21-schema-validator-ignores-dag-routing bug

**Second refactor (commit df43269, 2026-01-24):**
- Removed ALL schema validation from DAG layer entirely
- Moved validation to two-phase model:
  - **Phase 1:** Plugins self-validate during `__init__()` (plugin construction)
  - **Phase 2:** Cross-plugin compatibility checked during graph construction
- DAG validation now only checks structure (cycles, connectivity)
- Ref: `docs/plans/2026-01-24-fix-schema-validation-properly.md`

**3. Current architecture prevents this bug:**

The bug described early-exit behavior in `validate_pipeline_schemas()`:
```python
# OLD CODE (deleted):
if source_output is None:
    return errors  # ← Bug: skips all downstream validation
```

**Current architecture:**
- No standalone `validate_pipeline_schemas()` function exists
- No linear list-based validation exists
- Schema validation happens at graph construction time, edge-by-edge
- Dynamic schemas (None) handled per-edge, not globally:
  ```python
  # Current approach (from removed code in commit df43269):
  # For each edge:
  #   if producer_schema is None or consumer_schema is None:
  #     continue  # Skip THIS edge only, not all validation
  ```

**4. Related bugs also closed:**
- P1-2026-01-21-schema-validator-ignores-dag-routing: Closed (commit f4dd59d)
- P0-2026-01-24-eliminate-parallel-dynamic-schema-detection: Resolved via architecture refactor
- P0-2026-01-24-dynamic-schema-detection-regression: Superseded by root cause fix

### Why This is OBE (Not Just "Fixed")

This bug reported a defect in `validate_pipeline_schemas()` function. That function was:
1. First moved to `ExecutionGraph._validate_edge_schemas()` (f4dd59d)
2. Then deleted entirely when schema validation was moved to plugin construction (df43269)

The architecture changed so fundamentally that:
- The code path described in this bug no longer exists
- The validation model changed from linear to edge-based to plugin-based
- The problem cannot recur because the function is gone

**Status rationale:** OBE rather than FIXED because the fix wasn't a patch to the buggy code - the entire subsystem was replaced with a different architecture.

### Recommendation

**Move to:** `docs/bugs/closed/` (overtaken by events)

**Cross-references:**
- Implementation plan: `docs/plans/2026-01-24-fix-schema-validation-properly.md`
- Architecture review: `docs/bugs/arch-review-schema-validator-fix.md`
- Related closed bugs: P1-2026-01-21-schema-validator-ignores-dag-routing

---

## VERIFICATION: 2026-01-25

**Status:** OBE (CONFIRMED - Still Valid)

**Verified By:** Claude Code P2 verification wave 1

**Current Code Analysis:**

The previous verification from 2026-01-24 remains accurate. I have confirmed:

1. **Original buggy code deleted:** The `src/elspeth/engine/schema_validator.py` file with the `validate_pipeline_schemas()` function was deleted in commit `f4dd59d` (2026-01-24).

2. **Current architecture eliminates the bug pattern:**
   - Schema validation is now in `src/elspeth/core/dag.py` via `validate_edge_compatibility()` method
   - Validation is per-edge, not global, so dynamic schemas only bypass validation for THAT edge
   - Code at lines 699-701 shows the correct pattern:
     ```python
     # Rule 1: Dynamic schemas (None) bypass validation
     if producer_schema is None or consumer_schema is None:
         return  # Dynamic schema - compatible with anything
     ```
   - This returns from `_validate_single_edge()`, NOT from the entire validation loop
   - Other edges with explicit schemas continue to be validated

3. **Plugin-level validation added:**
   - All plugins now implement `validate_output_schema()` (Phase 1 self-validation)
   - Called during plugin `__init__()` before any graph construction
   - Verified in `src/elspeth/plugins/protocols.py` lines 106-116

**Git History:**

Recent relevant commits confirm the architecture change:
- `df43269` - refactor: remove schema validation from DAG layer
- `430307d` - feat: add schema validation to plugin protocols
- `7540e57` - fix: clarify validate_output_schema uses raise pattern, not list return
- `8809bd1` - feat: add edge compatibility validation to ExecutionGraph
- `7ee7c51` - feat: add self-validation to all builtin plugins

**Root Cause Confirmed:**

The root cause NO LONGER EXISTS. The original bug was:
```python
# OLD CODE (deleted in f4dd59d):
if source_output is None:
    return errors  # ← Bug: early exit from entire validation
```

Current code validates each edge independently:
```python
# CURRENT CODE (dag.py lines 657-658):
for from_id, to_id, _edge_data in self._graph.edges(data=True):
    self._validate_single_edge(from_id, to_id)
```

If an edge has dynamic schema, only THAT edge validation is skipped (line 701 returns from `_validate_single_edge()`, not from the loop).

**Recommendation:**

**CONFIRMED: Move to `docs/bugs/closed/` as OBE**

The bug is overtaken by events. The architecture changed from:
- Linear pipeline validation with global early-exit → Edge-based DAG validation with per-edge skipping
- Single-phase validation → Two-phase (plugin self-validation + edge compatibility)
- Centralized validator module → Distributed (plugins validate themselves, DAG validates edges)

This is a textbook OBE case: the problem cannot recur because the problematic code pattern no longer exists in the codebase.
