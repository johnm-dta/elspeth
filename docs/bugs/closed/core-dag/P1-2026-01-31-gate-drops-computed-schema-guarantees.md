# Bug Report: Gate nodes drop computed schema guarantees across pass-through

## Summary

- Gate nodes copy raw `config["schema"]` without preserving computed `output_schema_config` from upstream transforms, causing valid pipelines to fail DAG contract validation.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/dag.py:446-456` - gate nodes overwrite `node_config["schema"]` with the **raw** upstream schema config (`config["schema"]`).
- `src/elspeth/core/dag.py:460-472` - `output_schema_config` is only taken from the gate instance itself (typically `None`), so computed upstream schema config is not propagated.
- `src/elspeth/core/dag.py:1088-1122` - `_get_schema_config_from_node()` prefers `output_schema_config` when present, otherwise parses the raw `config["schema"]`.
- Because gate nodes lack `output_schema_config`, they fall back to raw schema and lose upstream computed guarantees (LLM `_output_schema_config`).

## Impact

- User-facing impact: Valid pipelines fail DAG validation with false contract violations
- Data integrity / security impact: None (fails safe, but incorrectly)
- Performance or cost impact: Developer time debugging false validation failures

## Root Cause Hypothesis

- When an LLM transform computes `output_schema_config` with additional guaranteed fields (like `*_usage`, `*_model`), gates only copy the raw schema config, not the computed schema.

## Proposed Fix

- Code changes:
  - Gates should inherit/pass through upstream's `output_schema_config` rather than raw config schema
  - Or: Gates with no schema should be transparent in contract validation
- Tests to add/update:
  - Add test: LLM transform -> gate -> downstream requiring LLM fields, assert validation passes

## Acceptance Criteria

- Gates correctly pass through upstream's computed guaranteed fields
- DAG validation passes for valid transform->gate->transform chains

## Resolution (2026-02-01)

**Status: FIXED**

### Root Cause Analysis

The bug was in `_get_effective_guaranteed_fields()` (lines 1201-1245). The method had a short-circuit at lines 1220-1222:

```python
own_guarantees = self._get_guaranteed_fields(node_id)
if own_guarantees:
    return own_guarantees
```

This short-circuit caused gates to return their raw schema guarantees (copied from upstream's `config["schema"]`) instead of walking upstream to find the computed `output_schema_config`. When the raw schema had *any* guarantees, the method would return early and never reach the pass-through logic at lines 1224-1243.

### The Fix

Modified `_get_effective_guaranteed_fields()` to check node type BEFORE attempting to use own guarantees:

1. **Gates ALWAYS inherit from upstream** - they don't compute schemas, so their raw schema is unreliable
2. **Coalesce nodes return intersection** of branch guarantees (unchanged)
3. **Other nodes return their own guarantees** (unchanged)

The key insight: Gates are pass-through nodes by definition. Their "own guarantees" from raw config are actually just an incomplete copy of upstream's config. The authoritative source is the upstream's `output_schema_config`.

### Tests Added

- `tests/core/test_dag_schema_propagation.py::TestGateSchemaConfigInheritance::test_gate_inherits_output_schema_config_from_upstream`
- `tests/core/test_dag_schema_propagation.py::TestGateSchemaConfigInheritance::test_chained_gates_inherit_through_all`

### Files Changed

- `src/elspeth/core/dag.py` - Fixed `_get_effective_guaranteed_fields()` to always walk upstream for gates
- `tests/core/test_dag_schema_propagation.py` - Added regression tests

### Verification

All 1267 core tests pass, including the new tests. The fix correctly propagates computed guarantees through gate nodes.
