# Bug Report: Gate routing errors leave node_state OPEN (MissingEdgeError / fork without TokenManager)

## Summary

- `GateExecutor.execute_gate()` begins a node_state and then performs routing resolution.
- If routing fails (MissingEdgeError) or fork occurs without a TokenManager, the method raises without completing the node_state, leaving an OPEN state in the audit trail.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/executors.py` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py`

## Steps To Reproduce

1. Configure a gate that returns a route label missing from the route resolution map, or trigger `fork_to_paths` with no TokenManager provided.
2. Run a pipeline that executes the gate.
3. Inspect the `node_states` table for the gate node.

## Expected Behavior

- The gate node_state is completed with status `failed` and an error payload when routing fails.

## Actual Behavior

- The node_state remains OPEN, with no completion record, even though execution failed.

## Evidence

- Node_state is opened before routing resolution:
  - `src/elspeth/engine/executors.py:355`
  - `src/elspeth/engine/executors.py:360`
- Missing edge raises without completing the state:
  - `src/elspeth/engine/executors.py:409`
  - `src/elspeth/engine/executors.py:411`
- Fork path raises without completing the state when no TokenManager:
  - `src/elspeth/engine/executors.py:432`
  - `src/elspeth/engine/executors.py:436`

## Impact

- User-facing impact: failures leave incomplete audit trails and complicate debugging.
- Data integrity / security impact: violates node_state completion invariants for failed executions.
- Performance or cost impact: none direct.

## Root Cause Hypothesis

- Routing errors occur after `begin_node_state()` but before any `complete_node_state()` call; exceptions are raised directly.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: wrap routing resolution in a try/except that records `complete_node_state(status="failed", error=...)` before raising.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests that simulate MissingEdgeError and missing TokenManager paths and assert failed node_state completion.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` (audit trail completeness).
- Observed divergence: failed gate routing leaves OPEN node_state.
- Reason (if known): exception paths bypass completion logic.
- Alignment plan or decision needed: define required node_state handling for routing errors.

## Acceptance Criteria

- Routing failures always result in a failed node_state record.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k gate_routing_error`
- New tests required: yes (gate routing failure audit).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
