# Bug Report: Engine supports plugin gates but ExecutionGraph.from_config does not build gate nodes/routes (route resolution mismatch)

## Summary

- The engine runtime (`RowProcessor`/`GateExecutor`) supports plugin gates (`BaseGate` / `GateProtocol`) and expects the execution graph to provide route resolution for gate route labels.
- `ExecutionGraph.from_config(...)` currently states that plugin-based gates were removed and builds only config gates, meaning plugin gate route resolution may be missing and can fail at runtime.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: pipelines using plugin gates in `row_plugins`
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: cross-check of engine gate support vs DAG builder behavior

## Steps To Reproduce

1. Add a plugin gate (subclass of `BaseGate`) to the transform pipeline (e.g., include it in `PipelineConfig.transforms` or in settings row_plugins if supported).
2. Build the graph via `ExecutionGraph.from_config(settings)`.
3. Run the pipeline and trigger a gate route label that should route to a sink.

## Expected Behavior

- If plugin gates are supported:
  - the graph contains gate nodes/edges for plugin gates and provides route resolution for gate labels, and routing events record against registered edges.
- If plugin gates are not supported:
  - the system fails fast at config/graph build time with a clear error (rather than failing mid-run).

## Actual Behavior

- Graph builder claims plugin gates were removed, but engine still executes them. Route resolution for plugin gate labels may be missing, causing runtime errors or incomplete routing.

## Evidence

- Engine runtime supports gates in the transform list:
  - `src/elspeth/engine/orchestrator.py:33` (`RowPlugin = BaseTransform | BaseGate`)
  - `src/elspeth/engine/processor.py:551` (`isinstance(transform, BaseGate)`)
- Graph builder explicitly says gates are config-driven only:
  - `src/elspeth/core/dag.py:292` (“Gate routing is now config-driven only … Plugin-based gates were removed”)
- GateExecutor requires route resolution for `RoutingKind.ROUTE`:
  - `src/elspeth/engine/executors.py:391`

## Impact

- User-facing impact: plugin-gate pipelines may break unexpectedly at runtime.
- Data integrity / security impact: missing route edges/resolution risks incorrect or missing routing event recording.
- Performance or cost impact: wasted runs and operator time.

## Root Cause Hypothesis

- Incremental transition from plugin gates to config gates left runtime support in place while graph compilation stopped emitting plugin gate routing metadata.

## Proposed Fix

- Decide and enforce one of:
  - Support plugin gates end-to-end:
    - update `ExecutionGraph.from_config` to create plugin gate nodes/edges and route maps, and ensure orchestrator assigns gate node_ids appropriately.
  - Remove plugin gate support:
    - fail fast if a `BaseGate` appears in `PipelineConfig.transforms` or settings row_plugins, with a migration path to config gates.
- Add tests that lock in the chosen behavior.

## Acceptance Criteria

- Plugin gate behavior is either fully supported (graph + engine) or explicitly rejected with a clear error.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_engine_gates.py`
  - `pytest tests/engine/test_orchestrator.py`
- New tests required: yes (graph compilation + runtime behavior)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`

## Verification (2026-01-25)

**Status: OBE (Overtaken By Events)**

This bug report is **obsolete**. The architectural concern it identified was valid at report time (2026-01-19) but was resolved through a deliberate refactoring that occurred around the same time period.

### What Changed

Between Jan 18-19, 2026, the codebase underwent a planned migration from plugin-based gates to config-driven gates:

1. **Plugin gate implementations deleted** (commit c9924af, 2026-01-18):
   - Removed FilterGate, FieldMatchGate, ThresholdGate plugin classes
   - Deleted `src/elspeth/plugins/gates/` directory entirely

2. **Plugin gate registration removed** (commit 01f0a96, 2026-01-18):
   - Removed `builtin_gates` registration from plugin manager
   - Gates became engine-level config operations

3. **Test suite migrated** (commits 0143ee5, c708db9, 06bbfad, 2e0ab07, 949f29f, 2026-01-19):
   - Converted all test BaseGate subclasses to GateSettings
   - One test verifying plugin+config gate interaction was added then immediately reverted (commits 9bc67a6, e9e95c0)

4. **Documentation updated** (commit in config.py):
   - Added comment: "Plugin-based gates were removed - use the gates: section instead"

### Current Architectural State

**BaseGate class is preserved but only for structural purposes:**

- **Defined**: `src/elspeth/plugins/base.py:185` - Defines the abstract base class with `evaluate()` method and node_id protocol
- **Engine support**: `src/elspeth/engine/processor.py:657` - Runtime still checks `isinstance(transform, BaseGate)`
- **Type alias**: `src/elspeth/engine/orchestrator.py:48` - `RowPlugin = BaseTransform | BaseGate`
- **Graph builder**: `src/elspeth/core/dag.py:374-390` - Treats all transforms uniformly (no special BaseGate handling)
- **Node ID assignment**: `src/elspeth/engine/orchestrator.py:397-405` - Plugin gates would get node_id if present in transforms list

**No plugin gate implementations exist:**

- No builtin plugins inherit from BaseGate
- No plugin gate discovery or registration
- All test gates migrated to GateSettings (config-driven)

**Config gates are the only gate implementation:**

- `src/elspeth/core/config.py:GateSettings` - Defines config-driven gate behavior
- `src/elspeth/core/dag.py:418-457` - Graph builder creates gate nodes from GateSettings only
- Route resolution map populated only for config gates (line 445, 448, 455)

### Why The Mismatch Was Tolerable

The "mismatch" identified in this bug report (engine supports BaseGate, graph doesn't build routes) is **by design**:

1. **BaseGate is infrastructure** - Kept for protocol definition, not for actual plugins
2. **No plugin gates to process** - All concrete gates were deleted, so the engine code path never executes
3. **DAG builder is correct** - Only builds routes for config gates because those are the only gates that exist
4. **Orchestrator would work** - If a BaseGate instance appeared in transforms, it would get a node_id and execute (though route resolution would fail at runtime since no routes exist for it in the graph)

### Is There Still A Problem?

**No runtime issue** - Since no plugin gates exist, the incomplete support cannot cause failures.

**Minor technical debt** - The BaseGate infrastructure remains in place with no concrete implementations. This could be cleaned up by either:
- Deleting BaseGate entirely (breaking change for anyone writing custom gates)
- Adding validation to reject BaseGate instances at config/graph build time
- Documenting BaseGate as "reserved for future use"

However, keeping the infrastructure has minimal cost and provides extension points if plugin gates are needed in the future.

### Recommendation

**Close as OBE.** The architectural concern was valid during the transition period but is no longer a bug given the current architecture. If the project wants to fully remove plugin gate support, that should be tracked as a separate technical debt cleanup item, not a bug fix.
