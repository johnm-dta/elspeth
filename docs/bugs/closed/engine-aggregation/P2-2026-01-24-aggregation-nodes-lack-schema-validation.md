# Bug Report: Aggregation nodes lack schema validation

## ✅ RESOLVED

**Status:** Fixed in RC-2 as part of P0-2026-01-24-schema-validation-non-functional
**Resolution:** Architectural refactor enables schema extraction from plugin instances
**Implementation:** See docs/plans/2026-01-24-schema-refactor-* files

This bug was a symptom of the broader P0 issue. The fix addresses all node types.

---

## ⚠️ SUPERSEDED BY P0-2026-01-24-schema-validation-non-functional

**This bug is a SYMPTOM of a broader architectural issue.**

Systematic debugging revealed that schema validation is completely non-functional for ALL node types (transforms, aggregations, gates, sources, sinks). The root cause is that the graph is built from config objects BEFORE plugins are instantiated, so schemas are never available.

**See:** `docs/bugs/P0-2026-01-24-schema-validation-non-functional.md` for the complete analysis and fix proposal.

**Implementation plan:** `docs/plans/2026-01-24-fix-schema-validation-architecture.md`

This bug report is retained for historical reference but should NOT be implemented in isolation. The fix must address the broader architectural issue.

---

## Summary (Original Report)

- Aggregation nodes are added to `ExecutionGraph` without `input_schema` or `output_schema` in `from_config()`
- Similar to gates, this causes edge validation to skip aggregation edges
- Unlike gates, aggregations have **dual schemas**: input schema (individual rows) ≠ output schema (batch result)
- Current schema inheritance approach (from gate fix) won't work - aggregations transform data, not pass-through

## Severity

- Severity: medium (less common than gates, but same validation gap)
- Priority: P2

## Reporter

- Name or handle: schema-inheritance-implementation
- Date: 2026-01-24
- Related run/issue ID: Follow-up from P1-2026-01-21-schema-validator-ignores-dag-routing

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-4` @ `88e977e`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any pipeline using aggregations

## Agent Context (if relevant)

- Goal or task prompt: Implement schema inheritance for gates
- Model/version: Claude Sonnet 4.5
- Tooling and permissions: full implementation access
- Determinism details: N/A
- Notable tool calls or steps: Identified during gate schema fix implementation

## Steps To Reproduce

1. Define a pipeline with an aggregation node (e.g., `BatchStats`)
2. Configure aggregation to accept rows with schema A (e.g., `{value: float, category: str}`)
3. Configure aggregation to output batch results with schema B (e.g., `{mean: float, count: int, category: str}`)
4. Configure sink after aggregation to require fields from schema B
5. Run schema validation

## Expected Behavior

- Validation checks edges INTO aggregation against `input_schema`
- Validation checks edges OUT OF aggregation against `output_schema`
- Incompatible schemas should cause validation failure

## Actual Behavior

- Aggregation nodes have no schemas when added via `from_config()`
- Validation skips all aggregation edges (both incoming and outgoing)
- Schema incompatibilities are not detected

## Technical Details

**Code location:** `src/elspeth/core/dag.py`, `from_config()` method (lines ~440-460)

**Current implementation:**
```python
graph.add_node(
    aid,
    node_type="aggregation",
    plugin_name=agg_config.plugin,
    config=agg_node_config,
)
# Note: NO input_schema or output_schema provided
```

**Challenge:** Aggregations have **dual schemas**:
- `input_schema`: Individual rows entering the aggregation
- `output_schema`: Batch result emitted after trigger fires

**Why gate fix doesn't work:**
- Gates are pass-through (inherit upstream schema)
- Aggregations **transform** data (input ≠ output)
- Need to handle both schemas separately in validation

## Root Cause (UPDATED - See P0 Bug)

**Original hypothesis:** Aggregation nodes were added after schema validation was implemented. The `from_config()` method doesn't extract schemas from aggregation config objects.

**Actual root cause (discovered via systematic debugging):**

Schema validation is non-functional for ALL node types. The architectural flaw:

1. Graph is built from `ElspethSettings` (config objects) via `ExecutionGraph.from_config(config)`
2. Schemas are attached to plugin INSTANCES in `__init__()` (e.g., `self.input_schema = ...`)
3. Plugins are instantiated AFTER graph construction in `_execute_pipeline()`
4. Therefore `getattr(plugin_config, "input_schema", None)` always returns `None`
5. Validation sees `None` and silently skips all edges

**Evidence:** `src/elspeth/cli.py` line 179 builds graph, line 373 instantiates plugins.

**Why this bug report only noticed aggregations:** The code reviewer saw missing `getattr()` calls for aggregations but didn't realize transforms have the same issue (just hidden by the `getattr()` returning `None`).

## Proposed Fix

**Phase 1: Schema Extraction**
1. Update `from_config()` to extract `input_schema` and `output_schema` from aggregation config
2. Add both schemas to aggregation nodes when constructing graph

**Phase 2: Dual Schema Validation**
1. Update `_validate_edge_schemas()` to handle aggregation nodes specially:
   - Edges INTO aggregation: validate against `input_schema`
   - Edges OUT OF aggregation: validate against `output_schema`
2. Check consumer node type when selecting schema

**Code changes:**
- Extend schema extraction in `from_config()` (similar to transforms)
- Add conditional in `_validate_edge_schemas()`:
  ```python
  # Get consumer schema (handles aggregation dual schemas)
  if to_info.node_type == "aggregation":
      consumer_schema = to_info.input_schema  # Incoming edge
  else:
      consumer_schema = to_info.input_schema  # Normal case
  ```

**Tests to add:**
- Pipeline with aggregation: source → aggregation → sink
- Verify incoming edge validates against `input_schema`
- Verify outgoing edge validates against `output_schema`
- Test incompatible schemas cause validation failure

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (compatibility between connected nodes)
- Observed divergence: Aggregation edges skip validation
- Reason: Aggregation config lacks schema extraction in `from_config()`
- Alignment plan: Extract schemas from aggregation config, handle dual schemas in validation

## Acceptance Criteria

- Aggregation nodes have both `input_schema` and `output_schema` when built via `from_config()`
- Edge validation checks incoming edges against `input_schema`
- Edge validation checks outgoing edges against `output_schema`
- Schema incompatibilities in aggregation pipelines are detected

## Tests

- Suggested tests to run: `pytest tests/core/test_dag.py -k aggregation`
- New tests required: yes

## Notes / Links

- **SUPERSEDED BY:** P0-2026-01-24-schema-validation-non-functional.md (broader architectural issue)
- **Implementation plan:** docs/plans/2026-01-24-fix-schema-validation-architecture.md
- Related issues/PRs: P1-2026-01-21-schema-validator-ignores-dag-routing (gate fix)
- Related issues/PRs: P3-2026-01-24-coalesce-nodes-lack-schema-validation (same root cause)
- Related design docs: `docs/contracts/plugin-protocol.md`
- Architecture review: `docs/bugs/arch-review-schema-validator-fix.md` (identified this issue)
- Investigation: Systematic debugging session (2026-01-24) + architecture-critic + python-code-reviewer agents

## Implementation Complexity

**Estimate:** Medium
- Requires dual schema handling (more complex than gates)
- Requires plugin protocol support (aggregations must define both schemas)
- Test coverage needs both incoming and outgoing edge cases

**Dependencies:**
- Aggregation plugins must define `input_schema` and `output_schema` attributes
- May require aggregation plugin updates if schemas not yet defined
