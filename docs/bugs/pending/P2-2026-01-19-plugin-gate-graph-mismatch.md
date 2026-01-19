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
