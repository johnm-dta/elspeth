# Bug Report: `elspeth run` validates one `ExecutionGraph` but executes another (unvalidated)

## Summary

- `elspeth run` builds and validates an `ExecutionGraph`, then calls `_execute_pipeline(config)` which **rebuilds the graph again** and executes with the second instance.
- `ExecutionGraph.from_config()` uses `uuid.uuid4()` in node IDs, so repeated builds produce different node IDs. This means:
  - validation does not apply to the exact graph instance passed to `Orchestrator.run()`
  - verbose/dry-run outputs are derived from a different graph than the one executed
  - extra work is done every run

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 1 (CLI), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/cli.py` and `src/elspeth/core/dag.py`

## Steps To Reproduce

1. Add a temporary debug print to log graph node IDs (or run in a debugger):
   - Build graph via `ExecutionGraph.from_config(settings)` twice.
2. Observe node IDs differ between builds because they embed random UUID hex.
3. In CLI:
   - `run()` builds/validates graph instance A.
   - `_execute_pipeline()` builds graph instance B and executes it without `graph.validate()`.

## Expected Behavior

- The graph instance that is validated should be the exact instance executed, or the execution path should validate the graph it will execute.

## Actual Behavior

- Validation occurs on one graph instance, then execution happens with a newly constructed, unvalidated instance.

## Evidence

- CLI validates a graph in `run()`:
  - `src/elspeth/cli.py:99-103`
- CLI executes by calling `_execute_pipeline(config)`:
  - `src/elspeth/cli.py:135`
- `_execute_pipeline()` rebuilds the graph and never calls `graph.validate()`:
  - `src/elspeth/cli.py:288-290`
- Node IDs embed random UUIDs:
  - `src/elspeth/core/dag.py:260-266`

## Impact

- User-facing impact: confusing debug/verbose output (graph-derived data doesn’t correspond to the executed run’s node IDs).
- Data integrity / security impact: low but non-zero risk if invariants are added to `graph.validate()` that are not fully enforced by `from_config()`.
- Performance or cost impact: redundant graph construction every run.

## Root Cause Hypothesis

- CLI was structured with separate “validate” and “execute” phases, but `_execute_pipeline()` remained responsible for graph construction for aggregation node mapping and orchestrator wiring.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`:
    - Change `_execute_pipeline()` signature to accept `graph: ExecutionGraph` (already built/validated).
    - In `run()`, build graph once, call `graph.validate()`, then pass it through.
    - Alternatively, move validation into `_execute_pipeline()` and remove the earlier build in `run()`.
- Config or schema changes: none.
- Tests to add/update:
  - Add a unit test that stubs `_execute_pipeline()` to assert it receives the validated graph instance (or that graph is only constructed once).
  - Add a test that `_execute_pipeline()` validates the graph before calling orchestrator.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: “validate then execute” should validate the executed plan.
- Reason (if known): incremental implementation of CLI runner.
- Alignment plan or decision needed: N/A

## Acceptance Criteria

- `elspeth run` constructs the execution graph once per invocation.
- The graph passed to `Orchestrator.run()` has been validated (and is the validated instance).

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py -k run`
- New tests required: yes (graph reuse/validation assertions)

## Notes / Links

- Related issues/PRs: N/A

---

## Resolution

**Status:** RESOLVED
**Resolved Date:** 2026-01-23
**Resolution Method:** Systematic debugging following superpowers:systematic-debugging skill

### Implementation

Changed `_execute_pipeline()` to accept the validated `graph` parameter instead of rebuilding it:

1. **src/elspeth/cli.py:222** - Pass `graph=graph` from `run()` to `_execute_pipeline()`
2. **src/elspeth/cli.py:320-336** - Updated `_execute_pipeline()` signature to require `graph: ExecutionGraph`
3. **src/elspeth/cli.py:383-385** - Removed redundant `ExecutionGraph.from_config()` call

### Tests Added

**tests/cli/test_run_command.py** - Added `TestRunCommandGraphReuse` class with 2 regression tests:

1. `test_run_constructs_graph_once` - Verifies `from_config()` called exactly once via mock
2. `test_validated_graph_has_consistent_node_ids` - Verifies DB node IDs match validated graph

### Verification

- ✅ All 17 CLI tests pass (15 existing + 2 new)
- ✅ Ruff linter passes with no issues
- ✅ Architectural review: 5/5 score from axiom-system-architect:architecture-critic
- ✅ Code review: Approved by pr-review-toolkit:code-reviewer

### Root Cause

`ExecutionGraph.from_config()` uses `uuid.uuid4().hex[:8]` for node IDs (dag.py:249), so each call produces different random IDs. The CLI was building graph twice: once for validation, once for execution.

### Pattern Alignment

This fix aligns with the existing `resume` command pattern (cli.py:1104) which reconstructs graph from database to preserve node IDs.
