# Bug Report: Aggregation nodes record hardcoded metadata instead of transform metadata

## Summary

- Aggregation nodes are registered with plugin_version="1.0.0" and determinism=DETERMINISTIC regardless of the actual batch-aware transform. This misrepresents non-deterministic aggregation transforms (e.g., LLM batch transforms) in the audit trail.

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
- Data set or fixture: any pipeline with aggregation using non-deterministic transform

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation node using a non-deterministic transform (e.g., azure_batch_llm).
2. Run a pipeline and inspect nodes table metadata for the aggregation node.

## Expected Behavior

- Aggregation node metadata should reflect the actual transform's plugin_version and determinism.

## Actual Behavior

- Aggregation nodes are registered as deterministic with a hardcoded version.

## Evidence

- Aggregation nodes are hardcoded in `src/elspeth/engine/orchestrator.py:569-573`.
- Aggregation transforms exist in config.transforms with real metadata, but are not used for node registration.

## Impact

- User-facing impact: audit metadata misrepresents LLM batch transforms as deterministic.
- Data integrity / security impact: audit trail accuracy compromised.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Node registration treats aggregation nodes as metadata-only, ignoring the actual transform instance that executes.

## Proposed Fix

- Code changes (modules/files):
  - Resolve aggregation node metadata from the batch-aware transform instance with matching node_id.
- Config or schema changes: N/A
- Tests to add/update:
  - Assert aggregation node determinism/version match transform metadata.
- Risks or migration steps:
  - Ensure aggregation transforms are discoverable via node_id during registration.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): audit trail must reflect actual plugin metadata.
- Observed divergence: aggregation nodes use placeholders.
- Reason (if known): node registration skips aggregation transform instances.
- Alignment plan or decision needed: define authoritative metadata source for aggregation nodes.

## Acceptance Criteria

- Aggregation nodes record determinism and plugin_version from their transform.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k aggregation -v`
- New tests required: yes, aggregation node metadata test.

## Notes / Links

- Related issues/PRs: P2-2026-01-15-node-metadata-hardcoded (config gates only)
- Related design docs: CLAUDE.md auditability standard

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 3

**Current Code Analysis:**

The bug is confirmed present in `/home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py` at lines 650-654:

```python
elif node_id in aggregation_node_ids:
    # Aggregations use batch-aware transforms - determinism depends on the transform
    # Default to deterministic (statistical operations are typically deterministic)
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
```

The comment acknowledges that "determinism depends on the transform" but then hardcodes both `plugin_version="1.0.0"` and `determinism=Determinism.DETERMINISTIC`.

**Root Cause Analysis:**

The issue stems from how aggregation nodes are mapped in the DAG layer:

1. **Aggregation transforms ARE instantiated** and have proper metadata (`determinism`, `plugin_version`)
   - Example: `azure_batch.py` has `determinism = Determinism.NON_DETERMINISTIC` (line 122)
   - All BaseTransform instances have these attributes with proper defaults

2. **Aggregation transforms ARE added to config.transforms** (see `/home/john/elspeth-rapid/src/elspeth/cli.py` lines 615-621)
   - They're appended to the transforms list after instantiation
   - They have their `node_id` attribute set

3. **But the graph treats aggregations separately:**
   - In `/home/john/elspeth-rapid/src/elspeth/core/dag.py` lines 374-391: Regular transforms are mapped via `transform_id_map[sequence] = node_id`
   - Lines 395-416: Aggregations get their own `aggregation_id_map[name] = node_id`
   - Aggregation transforms are NOT included in `transform_id_map`

4. **The orchestrator builds node_to_plugin mapping incorrectly:**
   - Lines 617-625: Only uses `transform_id_map`, `sink_id_map`, and source
   - Aggregation transforms are in `config.transforms` but their node_ids are NOT in `transform_id_map`
   - So aggregation node_ids are NOT in `node_to_plugin`
   - Falls back to hardcoded metadata at lines 650-654

**Evidence:**

- Aggregation transforms are accessible at runtime (line 1777 in orchestrator.py finds them by matching `node_id`)
- The transform instances have the correct metadata (verified in `azure_batch.py`, `openrouter.py`, etc.)
- The node_to_plugin mapping construction (lines 617-625) explicitly excludes aggregations per the comment at line 616
- No tests exist to validate aggregation node metadata matches transform metadata

**Git History:**

Commit 7144be3 (2026-01-15) "fix(engine): use actual plugin metadata for node registration" attempted to fix hardcoded metadata, but only addressed regular transforms, sources, and sinks. The commit did not handle aggregations.

No subsequent commits have addressed this issue.

**Root Cause Confirmed:**

Yes. The bug is present because:
1. Aggregation transforms have metadata but are not included in `node_to_plugin` mapping
2. The code falls back to hardcoded values for aggregation nodes
3. This misrepresents non-deterministic aggregations (LLM batch transforms) in the audit trail

**Recommendation:**

Keep open. This is a valid P2 audit integrity issue.

**Suggested Fix:**

The orchestrator needs to also map aggregation transforms into `node_to_plugin`. Since aggregation transforms are in `config.transforms` and have their `node_id` attribute set, the fix is:

```python
# After line 625, add:
for transform in config.transforms:
    if hasattr(transform, 'node_id') and transform.node_id in aggregation_node_ids:
        node_to_plugin[transform.node_id] = transform
```

Then remove the special case at lines 650-654 so aggregations use the normal metadata extraction path (lines 663-668).

**Test Required:**

A test verifying that when a non-deterministic aggregation transform (like azure_batch) is used, the registered node has:
- `determinism = Determinism.NON_DETERMINISTIC` (not DETERMINISTIC)
- `plugin_version` matching the transform's version (not "1.0.0")

---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Claude Opus 4.5

**Resolution:**

The fix was applied in `src/elspeth/engine/orchestrator.py` with two changes:

### 1. Add aggregation transforms to `node_to_plugin` mapping (lines 701-720)

Moved `aggregation_node_ids` creation earlier and updated the transform loop to include aggregation transforms:

```python
# Build node ID sets for special node types
config_gate_node_ids: set[NodeID] = set(config_gate_id_map.values())
aggregation_node_ids: set[NodeID] = set(aggregation_id_map.values())

# Map plugin instances to their node IDs for metadata extraction
# Config gates and coalesce nodes don't have plugin instances (they're structural)
# Aggregation transforms DO have instances - they're in config.transforms with node_id set
node_to_plugin: dict[NodeID, Any] = {}
if source_id is not None:
    node_to_plugin[source_id] = config.source
for seq, transform in enumerate(config.transforms):
    if seq in transform_id_map:
        # Regular transform - mapped by sequence number
        node_to_plugin[transform_id_map[seq]] = transform
    elif transform.node_id is not None and transform.node_id in aggregation_node_ids:
        # Aggregation transform - has node_id set by CLI, not in transform_id_map
        node_to_plugin[transform.node_id] = transform
```

### 2. Remove hardcoded aggregation fallback (lines 736-745)

Removed the `elif node_id in aggregation_node_ids` branch that hardcoded metadata. Aggregations now fall through to the normal metadata extraction path:

```python
# Config gates and coalesce nodes are structural (no plugin instances)
# Aggregations have plugin instances in node_to_plugin (transforms with metadata)
if node_id in config_gate_node_ids:
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
elif node_id in coalesce_node_ids:
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
else:
    # All other nodes (source, transforms, sinks, aggregations) use plugin metadata
    plugin = node_to_plugin[NodeID(node_id)]
    plugin_version = plugin.plugin_version
    determinism = plugin.determinism
```

**Test Added:**

New test `test_aggregation_node_uses_transform_metadata` in `tests/engine/test_orchestrator_audit.py`:
- Creates a non-deterministic batch transform with custom version "2.3.4"
- Creates an aggregation using that transform
- Verifies the aggregation node in Landscape has `plugin_version="2.3.4"` and `determinism=NON_DETERMINISTIC`

**Tests Passing:**

- All 36 aggregation and orchestrator audit tests pass
- New test specifically validates the fix

**Verified By:** Claude Opus 4.5 (2026-01-29)
