# Bug Report: Coalesce nodes lack schema validation

## ✅ RESOLVED

**Status:** Fixed in RC-2 as part of P0-2026-01-24-schema-validation-non-functional
**Resolution:** Architectural refactor enables schema extraction from plugin instances
**Implementation:** See docs/plans/2026-01-24-schema-refactor-* files

This bug was a symptom of the broader P0 issue. The fix addresses all node types.

---

## Summary

- Coalesce nodes are added to `ExecutionGraph` without `input_schema` or `output_schema` in `from_config()`
- Similar to gates and aggregations, this causes edge validation to skip coalesce edges
- Coalesce nodes merge results from parallel fork branches
- Output schema depends on merge strategy (union fields, nested structure, or field selection)

## Severity

- Severity: low (coalesce is advanced feature, less commonly used)
- Priority: P3

## Reporter

- Name or handle: schema-inheritance-implementation
- Date: 2026-01-24
- Related run/issue ID: Follow-up from P1-2026-01-21-schema-validator-ignores-dag-routing

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-4` @ `88e977e`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any pipeline using coalesce nodes (fork → branches → coalesce)

## Agent Context (if relevant)

- Goal or task prompt: Implement schema inheritance for gates
- Model/version: Claude Sonnet 4.5
- Tooling and permissions: full implementation access
- Determinism details: N/A
- Notable tool calls or steps: Identified during gate schema fix implementation

## Steps To Reproduce

1. Define a pipeline with fork gate creating parallel branches
2. Add coalesce node to merge branch results
3. Configure different schemas for each fork branch
4. Configure sink after coalesce expecting merged schema
5. Run schema validation

## Expected Behavior

- Validation checks edges INTO coalesce from all fork branches
- Validation checks edge OUT OF coalesce to downstream nodes
- Incompatible schemas should cause validation failure
- Merge strategy should determine output schema

## Actual Behavior

- Coalesce nodes have no schemas when added via `from_config()`
- Validation skips all coalesce edges (both incoming and outgoing)
- Schema incompatibilities in fork/coalesce patterns are not detected

## Technical Details

**Code location:** `src/elspeth/core/dag.py`, `from_config()` method (lines ~480-500)

**Current implementation:**
```python
graph.add_node(
    coalesce_id,
    node_type="coalesce",
    plugin_name="coalesce",
    config=coalesce_config,
)
# Note: NO input_schema or output_schema provided
```

**Challenge:** Coalesce output schema depends on merge strategy:
- **Union merge**: Output has all fields from all branches
- **Nested merge**: Output wraps each branch in namespace
- **Select merge**: Output has subset of fields

**Why gate fix doesn't work:**
- Gates are pass-through (single input schema)
- Coalesce **merges multiple inputs** with potentially different schemas
- Output schema is computed, not inherited

## Root Cause

Coalesce nodes were added after schema validation was implemented. The `from_config()` method doesn't compute schemas for coalesce nodes based on merge strategy.

## Proposed Fix

**Option 1: Dynamic Schema Computation**
1. Compute coalesce output schema based on merge strategy and input branch schemas
2. Validate all incoming branch schemas are compatible with merge strategy
3. Set computed `output_schema` on coalesce node

**Option 2: Skip Validation (Pragmatic)**
1. Document that coalesce nodes use dynamic schemas
2. Skip validation for coalesce edges (current behavior, but intentional)
3. Rely on runtime type checking in coalesce implementation

**Recommendation:** Option 2 (skip validation)
- Coalesce is advanced feature, typically followed by permissive output sink
- Schema computation is complex and merge-strategy-dependent
- Runtime validation in coalesce implementation is sufficient
- Lower priority than aggregations

**If implementing Option 1:**
```python
# Compute output schema based on merge strategy
if coalesce_config.strategy == "union":
    output_schema = _compute_union_schema(branch_schemas)
elif coalesce_config.strategy == "nested":
    output_schema = _compute_nested_schema(branch_schemas)
elif coalesce_config.strategy == "select":
    output_schema = _compute_select_schema(coalesce_config.fields)

graph.add_node(
    coalesce_id,
    node_type="coalesce",
    plugin_name="coalesce",
    config=coalesce_config,
    output_schema=output_schema,
)
```

**Tests to add (if implementing):**
- Fork → branches → coalesce → sink pipeline
- Verify coalesce output schema matches merge strategy
- Test incompatible schemas in union merge
- Test nested merge creates wrapped schema

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (compatibility between connected nodes)
- Observed divergence: Coalesce edges skip validation
- Reason: Coalesce config lacks schema computation in `from_config()`
- Alignment plan: Either compute schemas or document as intentionally dynamic

## Acceptance Criteria

**If implementing schema validation:**
- Coalesce nodes have `output_schema` computed from merge strategy
- Edge validation checks outgoing edges against computed schema
- Schema incompatibilities in coalesce pipelines are detected

**If documenting as dynamic:**
- Documentation explicitly states coalesce uses dynamic schemas
- Validation intentionally skips coalesce edges (not a bug)
- Runtime validation in coalesce implementation is sufficient

## Tests

- Suggested tests to run: `pytest tests/core/test_dag.py::TestCoalesceNodes`
- New tests required: yes (if implementing schema validation)

## Notes / Links

- Related issues/PRs: P1-2026-01-21-schema-validator-ignores-dag-routing (gate fix)
- Related issues/PRs: P2-2026-01-24-aggregation-nodes-lack-schema-validation (similar issue)
- Related design docs: `docs/contracts/plugin-protocol.md`
- Architecture review: `docs/bugs/arch-review-schema-validator-fix.md` (identified this issue)

## Implementation Complexity

**Estimate:** High (if implementing schema validation)
- Requires merge-strategy-aware schema computation
- Complex logic for union/nested/select strategies
- Multiple fork branches increase test surface area

**Estimate:** Low (if documenting as dynamic)
- Add documentation clarifying intentional behavior
- No code changes required

**Recommendation:** P3 priority suggests deferring until coalesce usage increases. Current behavior (skip validation) is safe since coalesce typically feeds permissive sinks.
