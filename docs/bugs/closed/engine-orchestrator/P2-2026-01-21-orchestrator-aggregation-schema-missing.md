# Bug Report: Aggregation nodes record dynamic schema instead of configured schema

## Summary

- Orchestrator reads schema from node_info.config["schema"], but aggregation nodes store plugin options under "options". As a result, aggregation nodes are registered with a dynamic schema even when an explicit schema was configured.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any aggregation config with explicit schema

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation with explicit schema under options (e.g., schema: {fields: strict}).
2. Run pipeline and inspect nodes table for aggregation node schema_config.

## Expected Behavior

- Aggregation node should record the configured schema from aggregation options.

## Actual Behavior

- Aggregation node is registered with schema {fields: dynamic}.

## Evidence

- Schema is taken from node_info.config["schema"] in `src/elspeth/engine/orchestrator.py:589-592`.
- Aggregation node config stores options under "options" in `src/elspeth/core/dag.py:309-324`.

## Impact

- User-facing impact: schema metadata in audit trail is wrong for aggregation nodes.
- Data integrity / security impact: audit trail cannot prove schema contract used for aggregation.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Aggregation node config wraps plugin options, so orchestrator misses schema nested under options.

## Proposed Fix

- Code changes (modules/files):
  - If node_type == aggregation, pull schema from node_info.config.get("options", {}).get("schema").
- Config or schema changes: N/A
- Tests to add/update:
  - Aggregation node schema recorded in nodes table matches configured schema.
- Risks or migration steps:
  - Ensure backward compatibility for aggregation configs without schema.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): audit trail should record configured schema per node.
- Observed divergence: aggregation schema recorded as dynamic.
- Reason (if known): schema lookup ignores options wrapper.
- Alignment plan or decision needed: define schema extraction rules for aggregation nodes.

## Acceptance Criteria

- Aggregation nodes register schema_config that matches aggregation options.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k aggregation -v`
- New tests required: yes, aggregation schema metadata test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 3

**Current Code Analysis:**

The bug is confirmed to still exist in the current codebase. The issue is a structural mismatch between how aggregation nodes store their configuration vs. how the orchestrator extracts schema_config.

**Evidence from current code:**

1. **Aggregation node config structure** (src/elspeth/core/dag.py:396-400):
   ```python
   agg_node_config = {
       "trigger": agg_config.trigger.model_dump(),
       "output_mode": agg_config.output_mode,
       "options": dict(agg_config.options),  # schema is nested here
   }
   ```
   Schema is stored at `config["options"]["schema"]`

2. **Orchestrator extraction** (src/elspeth/engine/orchestrator.py:672):
   ```python
   schema_dict = node_info.config.get("schema", {"fields": "dynamic"})
   ```
   Looking for schema at `config["schema"]` (top level)

3. **Real-world example** (examples/batch_aggregation/settings.yaml:30-31):
   ```yaml
   aggregations:
     - name: batch_totals
       options:
         schema:
           fields: dynamic
   ```

**Result:** Aggregation nodes are always registered with `schema_config = {"fields": "dynamic"}` regardless of their actual configured schema, because the orchestrator looks at the wrong path.

**Comparison with other node types:**

- Transforms: Store config directly from plugin instance (`transform.config`), which already has schema at top level
- Sources/Sinks: Similar pattern to transforms
- Aggregations: Unique wrapper structure with `trigger`, `output_mode`, and `options`

**Git History:**

Searched commits since 2026-01-21 affecting orchestrator.py:
- Commit f4dd59d (2026-01-24): "move schema validation into ExecutionGraph" - moved validation logic but did NOT fix this schema extraction bug
- Commit 22d7f96 (2026-01-24): Closed P2-2026-01-24-aggregation-nodes-lack-schema-validation - this is a DIFFERENT bug about edge schema validation, not audit trail schema_config recording

No commits have addressed this specific issue.

**Root Cause Confirmed:**

The orchestrator uses a generic schema extraction pattern that works for transforms, sources, and sinks (where config is the plugin's direct config object), but fails for aggregations where the config is a wrapper containing `trigger`, `output_mode`, and `options`.

**Impact Assessment:**

- **Audit trail integrity**: The nodes table records incorrect schema metadata for aggregations
- **Explain queries**: Users cannot see what schema contract was actually used for aggregation
- **Compliance risk**: Cannot prove to auditors what schema validation was applied to aggregated data
- **Severity**: Correctly categorized as P2 - this is metadata corruption in audit trail, not a functional failure

**Recommendation:**

Keep open. This is a valid P2 bug that should be fixed to maintain audit trail integrity. The fix requires special-casing aggregation nodes in the orchestrator's schema extraction logic (lines 670-673):

```python
# Aggregations wrap their options, other nodes have config directly
if node_info.node_type == "aggregation":
    schema_dict = node_info.config.get("options", {}).get("schema", {"fields": "dynamic"})
else:
    schema_dict = node_info.config.get("schema", {"fields": "dynamic"})
```

This is a straightforward fix with no architectural implications.

---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Commit `8bd4086` - "Commit all working tree changes" (2026-01-27)

**Resolution:**

The fix was applied in `src/elspeth/core/dag.py` by adding the schema at the top level of the aggregation node config, aligning it with other node types:

```python
agg_node_config = {
    "trigger": agg_config.trigger.model_dump(),
    "output_mode": agg_config.output_mode,
    "options": dict(agg_config.options),
    "schema": transform_config["schema"],  # Added - now at top level
}
```

This approach is cleaner than special-casing in the orchestrator because:
1. All node types now have consistent config structure with schema at top level
2. No conditional logic needed in orchestrator's schema extraction
3. The aggregation transform's schema is the authoritative source

**Verified By:** Claude Opus 4.5 (2026-01-29)
- Confirmed `agg_node_config` now includes `"schema": transform_config["schema"]` at line 512
- Orchestrator can now find schema via `node_info.config["schema"]` for all node types
